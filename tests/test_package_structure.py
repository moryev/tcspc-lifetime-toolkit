"""Issue-#3 relocation guards, not a second scientific validation suite.

Update responsibility paths deliberately when implementations move; retain the
old import list. New package initializers must stay minimal, not eagerly load
their entire subtree. Runtime checks complement the static import graph.
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

# Supported module import paths, not a promise about incidental/private names.
# Keep these through at least #12, even when a name.py becomes name/__init__.py.
OLD_MODULE_PATHS = (
    "models", "convolution", "forward_model", "irf", "irf_preparation",
    "irf_estimation", "measurements", "measurement_io", "experimental",
    "simulation", "datasets", "preprocessing", "features", "representations",
    "estimator_api", "baselines", "fitting", "classical_evaluation",
    "ml_models", "ml_evaluation", "cross_validation", "evaluation",
    "evaluation_results", "evaluation_core", "evaluation_uncertainty",
    "conditional_evaluation", "benchmark_plots", "mismatch_evaluation", "timing_evaluation",
    "generalization", "generalization_datasets", "generalization_evaluation",
    "irf_evaluation", "uncertainty_evaluation", "ml_uncertainty",
    "classical_uncertainty", "uncertainty_robustness", "bayesian",
    "bayesian_sampling", "bayesian_predictive", "bayesian_evaluation",
    "bayesian_mismatch_evaluation", "persistence", "config", "exceptions",
)

# Responsibility sets identify implementations, not their future spelling.
# Canonical implementations move; old paths and dependency restrictions stay.
EXTENSION = {f"{PACKAGE}.estimation.api"}
BASELINES = f"{PACKAGE}.estimation.baselines"
ESTIMATION = f"{PACKAGE}.estimation"
BAYESIAN = f"{ESTIMATION}.bayesian"
ML = f"{PACKAGE}.ml"
BENCHMARKS = f"{PACKAGE}.benchmarks"
GENERALIZATION_PROTOCOL = f"{BENCHMARKS}.generalization_protocol"
GENERALIZATION_DATASETS = f"{BENCHMARKS}.generalization_datasets"
GENERALIZATION = f"{BENCHMARKS}.generalization"
UNCERTAINTY = f"{PACKAGE}.uncertainty"
UNCERTAINTY_EVALUATION = f"{UNCERTAINTY}.evaluation"
UNCERTAINTY_EVALUATION_IMPORTS = {
    "dataclasses", "enum", "numpy", "numpy.typing", "scipy.stats",
    GENERALIZATION_PROTOCOL,
}
BENCHMARK_IMPORTS = {
    BENCHMARKS: set(),
    GENERALIZATION_PROTOCOL: {"dataclasses", "enum", "math"},
    GENERALIZATION_DATASETS: {
        "__future__", "dataclasses", "itertools", "numpy", "pandas", "numpy.typing",
        GENERALIZATION_PROTOCOL, f"{PACKAGE}.simulation.histograms",
    },
    GENERALIZATION: {
        "__future__", "dataclasses", "numpy", "pandas", "numpy.typing",
        "collections.abc", "typing", "sklearn.decomposition",
        f"{PACKAGE}.estimation.baselines", f"{PACKAGE}.estimation.classical",
        f"{PACKAGE}.config", f"{PACKAGE}.estimation.api",
        f"{PACKAGE}.evaluation.core", f"{PACKAGE}.evaluation.results",
        f"{PACKAGE}.preprocessing.features", GENERALIZATION_PROTOCOL,
        GENERALIZATION_DATASETS, f"{PACKAGE}.irf.profiles",
        f"{PACKAGE}.ml.evaluation", f"{PACKAGE}.ml.models",
        f"{PACKAGE}.preprocessing.representations",
    },
}
MIGRATED_SHIMS = {f"{PACKAGE}.{name}" for name in (
    "estimator_api", "baselines", "models", "convolution", "forward_model",
    "features", "representations", "irf_preparation", "irf_estimation",
    "measurements", "measurement_io", "experimental", "datasets",
    "fitting", "classical_evaluation", "bayesian", "bayesian_sampling", "bayesian_predictive",
    "ml_models", "ml_evaluation", "cross_validation",
    "evaluation_results", "evaluation_core", "evaluation_uncertainty",
    "conditional_evaluation", "benchmark_plots",
    "generalization", "generalization_datasets", "generalization_evaluation",
    "uncertainty_evaluation",
)}
# A package facade must run as a parent initializer, but canonical code must
# not explicitly import its old re-export surface as an implementation.
PREPROCESSING = f"{PACKAGE}.preprocessing"
IRF = f"{PACKAGE}.irf"
MEASUREMENT = f"{PACKAGE}.measurement"
SIMULATION = f"{PACKAGE}.simulation"
EVALUATION = f"{PACKAGE}.evaluation"
DIAGNOSTICS = f"{EVALUATION}.diagnostics"
DIAGNOSTIC_EXPORTS = (
    "generate_fitted_signal", "calculate_residuals", "calculate_reduced_residuals",
    "calculate_absolute_lifetime_error", "calculate_relative_lifetime_error",
    "calculate_poisson_deviance_residuals", "calculate_lifetime_errors",
)
DIAGNOSTIC_IMPORTS = {
    "numpy", "numpy.typing", f"{PACKAGE}.physics.models", f"{ESTIMATION}.fitting",
}
COMPATIBILITY_SURFACES = MIGRATED_SHIMS | {PREPROCESSING, IRF, SIMULATION, EVALUATION}
FACTUAL_IMPORTS = {
    f"{EVALUATION}.results": {
        "collections.abc", "dataclasses", "types", "typing", "numpy", "pandas", "numpy.typing",
    },
    f"{EVALUATION}.core": {
        "collections.abc", "numpy", "pandas", "numpy.typing", f"{EVALUATION}.results",
    },
    f"{EVALUATION}.attachments": {
        "dataclasses", "numpy", "pandas", f"{EVALUATION}.results",
    },
}
FACTUAL = set(FACTUAL_IMPORTS)
ANALYSIS_IMPORTS = {
    f"{EVALUATION}.conditional": {
        "__future__", "collections.abc", "numpy", "pandas", "numpy.typing",
        DIAGNOSTICS, f"{ESTIMATION}.classical", f"{EVALUATION}.core",
        f"{EVALUATION}.results", f"{ML}.evaluation",
    },
    f"{EVALUATION}.plots": {
        "__future__", "matplotlib.pyplot", "numpy", "pandas", "matplotlib.axes",
    },
}
PHYSICAL = {f"{PACKAGE}.{name}" for name in (
    "physics.models", "physics.convolution", "physics.forward_model", "irf.profiles",
)}
PHYSICS_IMPORTS = {
    f"{PACKAGE}.physics.models": {"numpy", "numpy.typing"},
    f"{PACKAGE}.physics.convolution": {"numpy", "numpy.typing", "scipy.signal"},
    f"{PACKAGE}.physics.forward_model": {
        "numpy", "numpy.typing", f"{PACKAGE}.physics.models",
        f"{PACKAGE}.physics.convolution", f"{PACKAGE}.irf.profiles",
    },
}
CLASSICAL_IMPORTS = {
    f"{ESTIMATION}.fitting": {
        "collections.abc", "dataclasses", "numpy", "numpy.typing", "scipy.optimize",
        f"{PACKAGE}.physics.forward_model", f"{PACKAGE}.physics.models",
    },
    f"{ESTIMATION}.classical": {
        "__future__", "dataclasses", "time", "pandas", "numpy", "numpy.typing",
        f"{PACKAGE}.physics.convolution", f"{PACKAGE}.physics.models",
        f"{PREPROCESSING}.histograms", f"{ESTIMATION}.fitting",
        DIAGNOSTICS,
    },
}
BAYESIAN_IMPORTS = {
    BAYESIAN: set(),
    f"{BAYESIAN}.posterior": {
        "__future__", "math", "dataclasses", "enum", "typing", "numpy", "numpy.typing",
        f"{PACKAGE}.exceptions", f"{IRF}.selection", f"{IRF}.profiles",
        f"{IRF}.preparation", f"{MEASUREMENT}.records", f"{ESTIMATION}.fitting",
        f"{PACKAGE}.physics.forward_model", f"{BAYESIAN}.sampling",
    },
    f"{BAYESIAN}.sampling": {
        "__future__", "math", "time", "warnings", "dataclasses", "enum", "typing",
        "numpy", "numpy.typing", "emcee", "emcee.autocorr",
        f"{BAYESIAN}.posterior", f"{PACKAGE}.physics.forward_model",
        f"{IRF}.profiles", f"{IRF}.preparation", f"{MEASUREMENT}.records",
    },
    f"{BAYESIAN}.predictive": {
        "__future__", "math", "time", "dataclasses", "numpy", "numpy.typing",
        f"{BAYESIAN}.posterior", f"{BAYESIAN}.sampling",
        f"{PACKAGE}.physics.forward_model", f"{SIMULATION}.histograms",
        DIAGNOSTICS,
    },
}
ML_IMPORTS = {
    ML: set(),
    f"{ML}.models": {
        "__future__", "numpy", "sklearn.decomposition", "sklearn.ensemble",
        "sklearn.linear_model", "sklearn.pipeline", "sklearn.preprocessing",
        f"{PACKAGE}.config", f"{ESTIMATION}.api", f"{PREPROCESSING}.representations",
    },
    f"{ML}.evaluation": {
        "dataclasses", "itertools", "numpy", "pandas", "numpy.typing", "typing",
        "sklearn.metrics", "sklearn.decomposition", "sklearn.model_selection",
        BASELINES, f"{PACKAGE}.config", f"{PREPROCESSING}.features",
        f"{PREPROCESSING}.representations", f"{SIMULATION}.histograms", f"{ML}.models",
        f"{EVALUATION}.core", f"{EVALUATION}.results",
    },
    f"{ML}.cross_validation": {
        "__future__", "collections.abc", "dataclasses", "typing", "numpy", "pandas",
        "numpy.typing", "sklearn.base", "sklearn.model_selection", f"{ML}.evaluation",
    },
}
SIMULATION_IMPORTS = {
    SIMULATION: {f"{SIMULATION}.histograms"},
    f"{SIMULATION}.histograms": {
        "numpy", "numpy.typing", f"{PACKAGE}.physics.convolution",
        f"{PACKAGE}.physics.models", f"{IRF}.profiles", f"{IRF}.preparation",
    },
    f"{SIMULATION}.datasets": {
        "dataclasses", "numpy", "numpy.typing", "pandas", f"{PACKAGE}.physics.models",
    },
}
PERSISTENCE = f"{PACKAGE}.persistence"
IRF_IMPORTS = {
    IRF: {f"{IRF}.profiles"},
    f"{IRF}.profiles": {
        "__future__", "copy", "dataclasses", "enum", "types", "typing",
        "numpy", "numpy.typing", "scipy.stats",
    },
    f"{IRF}.preparation": {
        "__future__", "dataclasses", "typing", "numpy", "numpy.typing",
        f"{IRF}.profiles", f"{MEASUREMENT}.records", f"{PREPROCESSING}.histograms",
    },
    f"{IRF}.estimation": {
        "__future__", "dataclasses", "typing", "numpy", "numpy.typing", "scipy.signal",
        f"{IRF}.profiles", f"{IRF}.preparation", f"{MEASUREMENT}.records",
        f"{PREPROCESSING}.histograms",
    },
    f"{IRF}.selection": {
        "__future__", "numpy", "numpy.typing", f"{PACKAGE}.exceptions",
        f"{IRF}.profiles", f"{IRF}.preparation", f"{MEASUREMENT}.records",
        f"{PREPROCESSING}.histograms",
    },
}
MEASUREMENT_IMPORTS = {
    MEASUREMENT: set(),
    f"{MEASUREMENT}.records": {
        "__future__", "copy", "dataclasses", "enum", "types", "typing", "numpy",
        "numpy.typing", f"{PACKAGE}.exceptions", f"{IRF}.profiles",
        f"{PREPROCESSING}.histograms",
    },
    f"{MEASUREMENT}.io": {
        "__future__", "csv", "decimal", "pathlib", "typing", "numpy",
        f"{PACKAGE}.exceptions", f"{MEASUREMENT}.records",
    },
    f"{MEASUREMENT}.adapters": {
        "__future__", "copy", "dataclasses", "types", "typing", "numpy", "pandas",
        "numpy.typing", f"{ESTIMATION}.classical", f"{PACKAGE}.config",
        DIAGNOSTICS, f"{PACKAGE}.exceptions",
        f"{PREPROCESSING}.features", f"{PREPROCESSING}.histograms",
        f"{IRF}.preparation", f"{IRF}.selection", f"{MEASUREMENT}.records",
    },
}
PREPROCESSING_IMPORTS = {
    PREPROCESSING: {f"{PREPROCESSING}.histograms"},
    f"{PREPROCESSING}.histograms": {
        "__future__", "logging", "numpy", "numpy.typing",
        f"{PACKAGE}.config", f"{PACKAGE}.exceptions",
    },
    f"{PREPROCESSING}.features": {
        "__future__", "numpy", "pandas", "numpy.typing",
        f"{PACKAGE}.config", f"{PACKAGE}.exceptions", f"{PREPROCESSING}.histograms",
    },
    f"{PREPROCESSING}.representations": {
        "__future__", "numpy", "numpy.typing", "sklearn.decomposition",
        f"{PACKAGE}.config", f"{PREPROCESSING}.histograms",
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
            typing_names.update(a.asname or a.name for a in node.names if a.name == "typing")
        elif isinstance(node, ast.ImportFrom) and node.module == "typing":
            checking_names.update(a.asname or a.name for a in node.names if a.name == "TYPE_CHECKING")
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
            checking = (
                isinstance(test, ast.Name) and test.id in checking_names
            ) or (
                isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
                and isinstance(test.value, ast.Name) and test.value.id in typing_names
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
                targets.update(p for p in packages if target.startswith(p + ".")
                               and not module.startswith(p + ".") and module != p)
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
        records[module] = _imports(path.read_text(encoding="utf-8"), module,
                                   is_package=is_package)
    return records, packages


def test_supported_flat_modules_remain_importable():
    for name in OLD_MODULE_PATHS:
        assert importlib.import_module(f"{PACKAGE}.{name}") is not None


def test_responsibility_dependency_direction(package_imports):
    records, _ = package_imports
    assert EXTENSION | FACTUAL | PHYSICAL <= records.keys()
    allowed_external = {
        "__future__", "collections", "dataclasses", "types", "typing", "numpy", "pandas",
    }
    for module in EXTENSION | FACTUAL:
        for _, base, names in records[module]:
            if base == PACKAGE or base.startswith(PACKAGE + "."):
                targets = _targets(base, names, set(records)) or {base}
                assert targets <= (FACTUAL if module in FACTUAL else set()), (module, targets)
            else:
                assert base.split(".")[0] in allowed_external, (module, base)
                if module in EXTENSION:
                    assert base.split(".")[0] != "pandas", (module, base)
    for module in PHYSICAL:
        for _, base, names in records[module]:
            if base == PACKAGE or base.startswith(PACKAGE + "."):
                assert (_targets(base, names, set(records)) or {base}) <= PHYSICAL, (module, base)
    for module, imports in records.items():
        if module == PERSISTENCE or module.startswith(PERSISTENCE + "."):
            continue
        for _, base, names in imports:
            targets = _targets(base, names, set(records)) | {base}
            assert not any(t == PERSISTENCE or t.startswith(PERSISTENCE + ".")
                           for t in targets), (module, targets)


def test_migrated_initializers_and_canonical_consumers_stay_independent(package_imports):
    records, _ = package_imports
    assert records[ESTIMATION] == []  # No eager, deferred, or typing imports.
    assert records[BAYESIAN] == []
    assert records[ML] == []
    assert records[f"{PACKAGE}.physics"] == []
    assert records[MEASUREMENT] == []
    # Preserve the existing DummyRegressor dependency, not a new algorithm.
    assert {base for _, base, _ in records[BASELINES]} == {
        "__future__", "numpy", "numpy.typing", "sklearn.dummy",
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
        kind == "eager" and base == f"{IRF}.profiles"
        for kind, base, _ in records[IRF]
    )
    # Records may depend on profiles, never on preparation via an aggregate
    # initializer: preparation itself consumes those canonical records.
    assert {base for _, base, _ in records[f"{MEASUREMENT}.records"]
            if base == IRF or base.startswith(IRF + ".")} == {f"{IRF}.profiles"}
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
                assert not targets.intersection({
                    f"{PACKAGE}.experimental", f"{MEASUREMENT}.adapters",
                }), (module, targets)


def test_benchmark_protocol_data_dependencies_preserve_existing_edge_kinds(package_imports):
    records, _ = package_imports
    tree = ast.parse((SOURCE_ROOT / "benchmarks" / "__init__.py").read_text(encoding="utf-8"))
    assert len(tree.body) == 1 and ast.get_docstring(tree)
    assert records[BENCHMARKS] == []
    for module, allowed in BENCHMARK_IMPORTS.items():
        assert {base for _, base, _ in records[module]} == allowed, module
        assert all(kind == "eager" for kind, _, _ in records[module]), module

    def benchmark_edges(module):
        return {(kind, base) for kind, base, _ in records[module]
                if base == BENCHMARKS or base.startswith(BENCHMARKS + ".")}

    assert benchmark_edges(f"{PACKAGE}.ml_uncertainty") == {
        ("typing", GENERALIZATION_PROTOCOL), ("typing", GENERALIZATION_DATASETS),
        ("typing", GENERALIZATION),
    }
    assert ("typing", GENERALIZATION,
            ("GeneralizationPreparedData",)) in records[f"{PACKAGE}.ml_uncertainty"]
    assert benchmark_edges(PERSISTENCE) == {
        ("deferred", GENERALIZATION_DATASETS), ("deferred", GENERALIZATION),
    }
    # Existing scientific provenance guard, not a protocol-independent metric layer.
    assert ("eager", GENERALIZATION_PROTOCOL, ("FINAL_ROBUSTNESS_TEST_IDS",)) in (
        records[UNCERTAINTY_EVALUATION]
    )
    assert benchmark_edges(GENERALIZATION) == {
        ("eager", GENERALIZATION_PROTOCOL), ("eager", GENERALIZATION_DATASETS),
    }
    assert benchmark_edges(f"{PACKAGE}.uncertainty_robustness") == {
        ("eager", GENERALIZATION_PROTOCOL), ("eager", GENERALIZATION_DATASETS),
        ("eager", GENERALIZATION),
    }
    consumers = {(module, kind, names) for module, imports in records.items()
                 for kind, base, names in imports if base == GENERALIZATION}
    facade = f"{PACKAGE}.generalization_evaluation"
    assert {(module, kind, names) for module, kind, names in consumers if module != facade} == {
        (f"{PACKAGE}.ml_uncertainty", "typing", ("GeneralizationPreparedData",)),
        (f"{PACKAGE}.uncertainty_robustness", "eager", ("GeneralizationPreparedData",)),
        (PERSISTENCE, "deferred", ("RobustnessMetrics",)),
    }
    assert {(module, kind) for module, kind, _ in consumers if module == facade} == {
        (facade, "eager"),
    }
    foundations = (
        f"{PACKAGE}.physics", IRF, MEASUREMENT, SIMULATION, PREPROCESSING, ESTIMATION,
    )
    for module in records:
        if (module in FACTUAL or module == f"{ML}.models"
                or any(module == p or module.startswith(p + ".") for p in foundations)):
            assert not benchmark_edges(module), module


@pytest.mark.parametrize("module", [
    BENCHMARKS, GENERALIZATION_PROTOCOL, GENERALIZATION_DATASETS,
    f"{PACKAGE}.generalization", f"{PACKAGE}.generalization_datasets",
])
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
    'tcspc_toolkit.uncertainty_evaluation', 'tcspc_toolkit.ml_uncertainty',
    'tcspc_toolkit.classical_uncertainty', 'tcspc_toolkit.uncertainty_robustness',
    'tcspc_toolkit.irf_evaluation', 'tcspc_toolkit.mismatch_evaluation',
    'tcspc_toolkit.timing_evaluation', 'tcspc_toolkit.bayesian_evaluation',
    'tcspc_toolkit.bayesian_mismatch_evaluation',
]
class BoundaryGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + '.') for p in blocked):
            attempts.append(fullname)
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, BoundaryGuard())
import tcspc_toolkit
assert not any(n == {BENCHMARKS!r} or n.startswith({BENCHMARKS + '.'!r}) for n in sys.modules)
namespace = importlib.import_module({BENCHMARKS!r})
assert not any(n.startswith({BENCHMARKS + '.'!r}) for n in sys.modules)
importlib.import_module({module!r})
expected = {{
    {BENCHMARKS!r}: set(),
    {GENERALIZATION_PROTOCOL!r}: {{{GENERALIZATION_PROTOCOL!r}}},
    {GENERALIZATION_DATASETS!r}: {{{GENERALIZATION_PROTOCOL!r}, {GENERALIZATION_DATASETS!r}}},
    'tcspc_toolkit.generalization': {{{GENERALIZATION_PROTOCOL!r}}},
    'tcspc_toolkit.generalization_datasets': {{{GENERALIZATION_PROTOCOL!r}, {GENERALIZATION_DATASETS!r}}},
}}[{module!r}]
assert {{n for n in sys.modules if n.startswith({BENCHMARKS + '.'!r})}} == expected
assert not hasattr(namespace, 'GeneralizationSuiteDefinition')
assert not hasattr(namespace, 'GeneralizationTestSuite')
assert not attempts, attempts
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("module", [
    GENERALIZATION, f"{PACKAGE}.generalization_evaluation",
])
def test_fresh_generalization_execution_uses_only_canonical_dependencies(tmp_path, module):
    script = f"""
import importlib
import importlib.abc
import sys
attempts = []
blocked = {sorted(MIGRATED_SHIMS - {f"{PACKAGE}.generalization_evaluation"})!r} + [
    'emcee', 'sqlite3', 'tcspc_toolkit.persistence',
    'tcspc_toolkit.estimation.bayesian', 'tcspc_toolkit.uncertainty_evaluation',
    'tcspc_toolkit.ml_uncertainty', 'tcspc_toolkit.classical_uncertainty',
    'tcspc_toolkit.uncertainty_robustness', 'tcspc_toolkit.irf_evaluation',
    'tcspc_toolkit.mismatch_evaluation', 'tcspc_toolkit.timing_evaluation',
    'tcspc_toolkit.bayesian_evaluation', 'tcspc_toolkit.bayesian_mismatch_evaluation',
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
assert not any(n.startswith({BENCHMARKS + '.'!r}) for n in sys.modules)
importlib.import_module({module!r})
assert {{n for n in sys.modules if n.startswith({BENCHMARKS + '.'!r})}} == {{
    {GENERALIZATION!r}, {GENERALIZATION_PROTOCOL!r}, {GENERALIZATION_DATASETS!r},
}}
assert not hasattr(namespace, 'GeneralizationPreparedData')
assert tuple(tcspc_toolkit.__all__) == exports
assert not attempts, attempts
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


def test_fresh_typing_and_deferred_consumers_do_not_load_generalization_execution(tmp_path):
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
for module in ('tcspc_toolkit.ml_uncertainty', 'tcspc_toolkit.persistence'):
    importlib.import_module(module)
assert not attempts, attempts
assert not set(blocked).intersection(sys.modules)
"""
    completed = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("filename, canonical, count", [
    ("generalization_evaluation.py", GENERALIZATION, 61),
    ("uncertainty_evaluation.py", UNCERTAINTY_EVALUATION, 19),
])
def test_study_facades_contain_only_direct_compatibility_exports(filename, canonical, count):
    path = SOURCE_ROOT / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert ast.get_docstring(tree)
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
    assert len(tree.body) == 3 and len(imports) == 1
    assert imports[0].module == canonical and imports[0].level == 0
    assert all(alias.asname is None and not alias.name.startswith("_") for alias in imports[0].names)
    exports = tree.body[-1]
    assert isinstance(exports, ast.Assign) and len(exports.targets) == 1
    assert exports.targets[0].id == "__all__"
    assert ast.literal_eval(exports.value) == tuple(alias.name for alias in imports[0].names)
    assert len(imports[0].names) == count


def test_uncertainty_evaluation_retains_protocol_policy_and_consumer_edge_kinds(package_imports):
    records, _ = package_imports
    tree = ast.parse((SOURCE_ROOT / "uncertainty" / "__init__.py").read_text(encoding="utf-8"))
    assert len(tree.body) == 1 and ast.get_docstring(tree)
    assert records[UNCERTAINTY] == []
    assert {base for _, base, _ in records[UNCERTAINTY_EVALUATION]} == UNCERTAINTY_EVALUATION_IMPORTS
    assert all(kind == "eager" for kind, _, _ in records[UNCERTAINTY_EVALUATION])
    internal = [(kind, base, names) for kind, base, names in records[UNCERTAINTY_EVALUATION]
                if base.startswith(PACKAGE + ".")]
    assert internal == [("eager", GENERALIZATION_PROTOCOL, ("FINAL_ROBUSTNESS_TEST_IDS",))]
    consumers = [(module, kind, names) for module, imports in records.items()
                 for kind, base, names in imports if base == UNCERTAINTY_EVALUATION]
    assert {(module, kind) for module, kind, _ in consumers} == {
        (f"{PACKAGE}.uncertainty_evaluation", "eager"),
        (f"{PACKAGE}.ml_uncertainty", "eager"),
        (f"{PACKAGE}.classical_uncertainty", "eager"),
        (f"{PACKAGE}.uncertainty_robustness", "eager"),
        (PERSISTENCE, "deferred"),
    }
    assert len(consumers) == 12  # Facade, three eager consumers, eight local imports.
    assert sum(module == PERSISTENCE for module, _, _ in consumers) == 8
    foundations = (
        f"{PACKAGE}.physics", IRF, MEASUREMENT, SIMULATION, PREPROCESSING, ESTIMATION,
    )
    for module, imports in records.items():
        if (module in FACTUAL or module == f"{ML}.models"
                or any(module == p or module.startswith(p + ".") for p in foundations)):
            for _, base, names in imports:
                targets = _targets(base, names, set(records)) | {base}
                assert not any(t == UNCERTAINTY or t.startswith(UNCERTAINTY + ".")
                               or t == f"{PACKAGE}.uncertainty_evaluation" for t in targets), module


@pytest.mark.parametrize("module", [
    UNCERTAINTY, UNCERTAINTY_EVALUATION, f"{PACKAGE}.uncertainty_evaluation",
])
def test_fresh_uncertainty_imports_preserve_initializer_and_protocol_only_boundary(tmp_path, module):
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
    'tcspc_toolkit.estimation.bayesian', 'tcspc_toolkit.ml_uncertainty',
    'tcspc_toolkit.classical_uncertainty', 'tcspc_toolkit.uncertainty_robustness',
    'tcspc_toolkit.evaluation.attachments', {GENERALIZATION!r}, {GENERALIZATION_DATASETS!r},
    'tcspc_toolkit.irf_evaluation', 'tcspc_toolkit.mismatch_evaluation',
    'tcspc_toolkit.timing_evaluation', 'tcspc_toolkit.bayesian_evaluation',
    'tcspc_toolkit.bayesian_mismatch_evaluation',
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
assert not any(n == {UNCERTAINTY!r} or n.startswith({UNCERTAINTY + '.'!r}) for n in sys.modules)
assert not any(n == {BENCHMARKS!r} or n.startswith({BENCHMARKS + '.'!r}) for n in sys.modules)
inherited = set(sys.modules)
original_import = builtins.__import__
seen = {{}}
def track(name, globals=None, locals=None, fromlist=(), level=0):
    caller = (globals or {{}}).get('__name__', '')
    if caller == {UNCERTAINTY!r} or caller.startswith({UNCERTAINTY + '.'!r}):
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
    assert {{n for n in sys.modules if n.startswith({BENCHMARKS + '.'!r})}} == {{
        {GENERALIZATION_PROTOCOL!r},
    }}
assert not any(n == 'sklearn' or n.startswith('sklearn.') for n in set(sys.modules) - inherited)
assert not hasattr(namespace, 'PredictionIntervalResult')
assert not hasattr(namespace, 'IntervalAttachment')
assert tuple(tcspc_toolkit.__all__) == root_exports
assert not attempts, attempts
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
                               capture_output=True, text=True, timeout=60)
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
    completed = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


def test_preprocessing_collision_has_one_package_implementation():
    assert not (SOURCE_ROOT / "preprocessing.py").exists()
    package = importlib.import_module(PREPROCESSING)
    assert Path(package.__file__).resolve() == SOURCE_ROOT / "preprocessing" / "__init__.py"
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
    for name in ("PreparedIRF", "prepare_irf", "LeadingEdgeIRFResult",
                 "estimate_irf_from_leading_edge", "_sampled_fwhm", "_resolve_reconvolution_irf"):
        assert not hasattr(package, name), name


def test_simulation_collision_preserves_only_the_historical_histogram_surface():
    assert not (SOURCE_ROOT / "simulation.py").exists()
    package = importlib.import_module(SIMULATION)
    assert Path(package.__file__).resolve() == SOURCE_ROOT / "simulation" / "__init__.py"
    assert hasattr(package, "__path__")
    importlib.import_module(f"{SIMULATION}.datasets")
    for name in ("SyntheticDataset", "generate_monoexponential_dataset",
                 "generate_grouped_monoexponential_dataset", "monoexponential_decay"):
        assert not hasattr(package, name), name


def test_evaluation_collision_preserves_only_the_historical_diagnostics_surface():
    assert not (SOURCE_ROOT / "evaluation.py").exists()
    package = importlib.import_module(EVALUATION)
    canonical = importlib.import_module(DIAGNOSTICS)
    root = importlib.import_module(PACKAGE)
    assert Path(package.__file__).resolve() == SOURCE_ROOT / "evaluation" / "__init__.py"
    assert hasattr(package, "__path__")
    assert tuple(package.__all__) == DIAGNOSTIC_EXPORTS
    assert not set(DIAGNOSTIC_EXPORTS).intersection(root.__all__)
    for name in DIAGNOSTIC_EXPORTS:
        assert getattr(package, name) is getattr(canonical, name)
        assert getattr(canonical, name).__module__ == DIAGNOSTICS
    for name in ("LifetimeFitResult", "monoexponential_decay", "np", "NDArray",
                 "PointEvaluationResult", "build_point_evaluation", "IntervalAttachment",
                 "build_prediction_diagnostics", "plot_true_vs_predicted"):
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
    'tcspc_toolkit.uncertainty_evaluation', 'tcspc_toolkit.ml_uncertainty',
    'tcspc_toolkit.classical_uncertainty', 'tcspc_toolkit.uncertainty_robustness',
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
    completed = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("module", sorted(FACTUAL_IMPORTS))
def test_fresh_factual_imports_separate_source_dependencies_from_parent_cost(tmp_path, module):
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
    'tcspc_toolkit.mismatch_evaluation', 'tcspc_toolkit.uncertainty_evaluation',
    'tcspc_toolkit.ml_uncertainty', 'tcspc_toolkit.classical_uncertainty',
    'tcspc_toolkit.uncertainty_robustness',
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
    if caller == {EVALUATION!r} or caller.startswith({EVALUATION + '.'!r}):
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
expected = {{{module!r}, {EVALUATION + '.results'!r}}}
assert set(seen) == expected | {{{EVALUATION!r}, {DIAGNOSTICS!r}}}
for caller in expected:
    assert seen[caller] == {FACTUAL_IMPORTS!r}[caller], (caller, seen[caller])
assert tuple(sys.modules[{EVALUATION!r}].__all__) == {DIAGNOSTIC_EXPORTS!r}
assert not attempts, attempts
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("module, canonical", [
    (f"{EVALUATION}.conditional", f"{EVALUATION}.conditional"),
    (f"{PACKAGE}.conditional_evaluation", f"{EVALUATION}.conditional"),
    (f"{EVALUATION}.plots", f"{EVALUATION}.plots"),
    (f"{PACKAGE}.benchmark_plots", f"{EVALUATION}.plots"),
])
def test_fresh_analysis_imports_require_explicit_submodules(tmp_path, module, canonical):
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
    'tcspc_toolkit.mismatch_evaluation', 'tcspc_toolkit.uncertainty_evaluation',
    'tcspc_toolkit.ml_uncertainty', 'tcspc_toolkit.classical_uncertainty',
    'tcspc_toolkit.uncertainty_robustness',
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
if {canonical!r} == {EVALUATION + '.conditional'!r}:
    assert {EVALUATION + '.plots'!r} not in sys.modules
    assert not any(n == 'matplotlib' or n.startswith('matplotlib.') for n in sys.modules)
else:
    assert 'matplotlib.pyplot' in sys.modules
    assert {EVALUATION + '.conditional'!r} not in sys.modules
    assert not any(n in sys.modules for n in {sorted(FACTUAL)!r})
assert not attempts, attempts
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
legacy = importlib.import_module(
    'tcspc_toolkit.conditional_evaluation'
    if {canonical!r} == {EVALUATION + '.conditional'!r} else 'tcspc_toolkit.benchmark_plots'
)
for name in legacy.__all__:
    assert getattr(legacy, name) is getattr(implementation, name)
    assert not hasattr(facade, name)
assert tuple(facade.__all__) == {DIAGNOSTIC_EXPORTS!r}
"""
    completed = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
                               capture_output=True, text=True, timeout=60)
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
        module: {(kind, base) for kind, base, _ in records[module]
                 if base.startswith(BAYESIAN + ".")}
        for module in BAYESIAN_IMPORTS
    }
    assert internal == {
        BAYESIAN: set(),
        f"{BAYESIAN}.posterior": {("deferred", f"{BAYESIAN}.sampling")},
        f"{BAYESIAN}.sampling": {("eager", f"{BAYESIAN}.posterior")},
        f"{BAYESIAN}.predictive": {
            ("eager", f"{BAYESIAN}.posterior"), ("eager", f"{BAYESIAN}.sampling"),
        },
    }
    sampler_edges = [(module, kind, base) for module in BAYESIAN_IMPORTS
                     for kind, base, _ in records[module]
                     if base == "emcee" or base.startswith("emcee.")]
    assert sampler_edges
    assert all(module == f"{BAYESIAN}.sampling" and kind == "deferred"
               for module, kind, _ in sampler_edges)


@pytest.mark.parametrize("module", sorted(BAYESIAN_IMPORTS))
def test_fresh_bayesian_import_preserves_optional_sampler_and_minimal_initializers(tmp_path, module):
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
assert not any(n == {BAYESIAN!r} or n.startswith({BAYESIAN + '.'!r}) for n in sys.modules)
importlib.import_module({module!r})
expected = {{
    {BAYESIAN!r}: set(),
    {BAYESIAN + '.posterior'!r}: {{'posterior'}},
    {BAYESIAN + '.sampling'!r}: {{'posterior', 'sampling'}},
    {BAYESIAN + '.predictive'!r}: {{'posterior', 'sampling', 'predictive'}},
}}[{module!r}]
loaded = {{n.rsplit('.', 1)[-1] for n in sys.modules if n.startswith({BAYESIAN + '.'!r})}}
assert loaded == expected, loaded
assert not attempts, attempts
blocked = {sorted(MIGRATED_SHIMS)!r} + [
    'emcee', 'sqlite3', 'tcspc_toolkit.persistence', 'tcspc_toolkit.bayesian_evaluation',
    'tcspc_toolkit.bayesian_mismatch_evaluation', 'tcspc_toolkit.generalization_evaluation',
    'tcspc_toolkit.uncertainty_robustness',
]
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("module", sorted(ML_IMPORTS))
def test_fresh_ml_import_preserves_minimal_initializer_and_dependency_direction(tmp_path, module):
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
assert not any(n == {ML!r} or n.startswith({ML + '.'!r}) for n in sys.modules)
original_import = builtins.__import__
seen = {{}}
def track(name, globals=None, locals=None, fromlist=(), level=0):
    caller = (globals or {{}}).get('__name__', '')
    if caller == {ML!r} or caller.startswith({ML + '.'!r}):
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
    {ML + '.models'!r}: {{'models'}},
    {ML + '.evaluation'!r}: {{'models', 'evaluation'}},
    {ML + '.cross_validation'!r}: {{'models', 'evaluation', 'cross_validation'}},
}}[{module!r}]
loaded = {{n.rsplit('.', 1)[-1] for n in sys.modules if n.startswith({ML + '.'!r})}}
assert loaded == expected, loaded
blocked = {sorted(MIGRATED_SHIMS)!r} + [
    {BAYESIAN!r}, 'emcee', 'sqlite3', 'tcspc_toolkit.persistence',
    'tcspc_toolkit.generalization_evaluation', 'tcspc_toolkit.ml_uncertainty',
    'tcspc_toolkit.uncertainty_robustness',
]
assert not attempts, attempts
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
                               capture_output=True, text=True, timeout=60)
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
        "demo.leaf": ("""
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
""", False),
        "demo.eager": ("", False),
        "demo.local": ("from . import leaf", False),
        "demo.typed": ("from . import leaf", False),
        "outside": ("import demo.leaf", False),
    }
    records = {name: _imports(source, name, is_package=package)
               for name, (source, package) in sources.items()}
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
def test_fresh_import_order_without_sampler_or_database(tmp_path, reverse, generic_only):
    names = (["estimation.api", "estimator_api", "evaluation_results", "evaluation_core", "evaluation_uncertainty",
              "evaluation.results", "evaluation.core", "evaluation.attachments"]
             if generic_only else [*OLD_MODULE_PATHS, "estimation.api", "estimation.baselines",
                                   "estimation.fitting", "estimation.classical",
                                   "estimation.bayesian.posterior", "estimation.bayesian.sampling",
                                   "estimation.bayesian.predictive",
                                   "ml.models", "ml.evaluation", "ml.cross_validation",
                                   "physics.models", "physics.convolution", "physics.forward_model",
                                   "preprocessing.histograms", "preprocessing.features",
                                   "preprocessing.representations",
                                   "irf.profiles", "irf.preparation", "irf.estimation", "irf.selection",
                                   "measurement.records", "measurement.io", "measurement.adapters",
                                   "simulation.histograms", "simulation.datasets",
                                   "evaluation.diagnostics", "evaluation.results",
                                   "evaluation.core", "evaluation.attachments",
                                   "evaluation.conditional", "evaluation.plots",
                                   "benchmarks.generalization_protocol",
                                   "benchmarks.generalization_datasets",
                                   "benchmarks.generalization",
                                   "uncertainty", "uncertainty.evaluation"])
    if reverse:
        names.reverse()
    blocked = ["emcee"]
    if generic_only:
        blocked += [PERSISTENCE, f"{PACKAGE}.bayesian_sampling", BAYESIAN, ML, BENCHMARKS,
                    f"{PACKAGE}.generalization_evaluation", f"{PACKAGE}.ml_models"]
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
    completed = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
                               capture_output=True, text=True, timeout=60)
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
    if caller == {PREPROCESSING!r} or caller.startswith({PREPROCESSING + '.'!r}):
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
assert seen[{PREPROCESSING!r}] == {{{PREPROCESSING + '.histograms'!r}}}
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
    'tcspc_toolkit.ml_models', 'tcspc_toolkit.uncertainty_robustness',
]
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
                               capture_output=True, text=True, timeout=60)
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
    if caller == {IRF!r} or caller.startswith({IRF + '.'!r}):
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
assert seen[{IRF!r}] == {{{IRF + '.profiles'!r}}}
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
    'tcspc_toolkit.ml_models', 'tcspc_toolkit.uncertainty_robustness',
]
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
                               capture_output=True, text=True, timeout=60)
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
    if caller == {MEASUREMENT!r} or caller.startswith({MEASUREMENT + '.'!r}):
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
    'tcspc_toolkit.generalization_evaluation', 'tcspc_toolkit.uncertainty_robustness',
]
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
                               capture_output=True, text=True, timeout=60)
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
    if caller == {SIMULATION!r} or caller.startswith({SIMULATION + '.'!r}):
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
assert seen[{SIMULATION!r}] == {{{SIMULATION + '.histograms'!r}}}
allowed = {SIMULATION_IMPORTS!r}
for caller, imports in seen.items():
    assert imports <= allowed[caller], (caller, imports)
from tcspc_toolkit.simulation import sample_photon_counts
from tcspc_toolkit.simulation.histograms import sample_photon_counts as canonical_sample
assert sample_photon_counts is canonical_sample
blocked = {sorted(MIGRATED_SHIMS)!r} + [
    'tcspc_toolkit.persistence', 'emcee', 'tcspc_toolkit.bayesian',
    'tcspc_toolkit.generalization_evaluation', 'tcspc_toolkit.ml_models',
    'tcspc_toolkit.uncertainty_robustness',
]
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
                               capture_output=True, text=True, timeout=60)
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
    'tcspc_toolkit.ml_models', 'tcspc_toolkit.uncertainty_robustness',
]
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("module", ["api", "baselines", "fitting", "classical"])
def test_fresh_canonical_estimation_import_adds_only_its_own_dependencies(tmp_path, module):
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
assert not any(n == {BAYESIAN!r} or n.startswith({BAYESIAN + '.'!r}) for n in sys.modules)
assert 'tcspc_toolkit.generalization_evaluation' not in sys.modules
assert not any(n == {ML!r} or n.startswith({ML + '.'!r}) for n in sys.modules)
assert 'tcspc_toolkit.uncertainty_robustness' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []
