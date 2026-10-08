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
    "conditional_evaluation", "mismatch_evaluation", "timing_evaluation",
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
MIGRATED_SHIMS = {f"{PACKAGE}.{name}" for name in (
    "estimator_api", "baselines", "models", "convolution", "forward_model",
    "features", "representations", "irf_preparation", "irf_estimation",
    "measurements", "measurement_io", "experimental",
)}
# A package facade must run as a parent initializer, but canonical code must
# not explicitly import its old re-export surface as an implementation.
PREPROCESSING = f"{PACKAGE}.preprocessing"
IRF = f"{PACKAGE}.irf"
MEASUREMENT = f"{PACKAGE}.measurement"
COMPATIBILITY_SURFACES = MIGRATED_SHIMS | {PREPROCESSING, IRF}
FACTUAL = {f"{PACKAGE}.{name}" for name in (
    "evaluation_results", "evaluation_core", "evaluation_uncertainty",
)}
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
        "numpy.typing", f"{PACKAGE}.classical_evaluation", f"{PACKAGE}.config",
        f"{PACKAGE}.evaluation", f"{PACKAGE}.exceptions",
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
    assert records[f"{PACKAGE}.physics"] == []
    assert records[MEASUREMENT] == []
    # Preserve the existing DummyRegressor dependency, not a new algorithm.
    assert {base for _, base, _ in records[BASELINES]} == {
        "__future__", "numpy", "numpy.typing", "sklearn.dummy",
    }
    for module, allowed in PHYSICS_IMPORTS.items():
        assert {base for _, base, _ in records[module]} <= allowed, module
    for module, allowed in PREPROCESSING_IMPORTS.items():
        assert {base for _, base, _ in records[module]} <= allowed, module
    for module, allowed in IRF_IMPORTS.items():
        assert {base for _, base, _ in records[module]} <= allowed, module
    for module, allowed in MEASUREMENT_IMPORTS.items():
        assert {base for _, base, _ in records[module]} <= allowed, module
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
    for module, imports in records.items():
        for _, base, names in imports:
            targets = _targets(base, names, set(records)) | {base}
            assert not targets.intersection(COMPATIBILITY_SURFACES), (module, targets)
            if module.startswith(f"{PACKAGE}.bayesian"):
                assert not targets.intersection({
                    f"{PACKAGE}.experimental", f"{MEASUREMENT}.adapters",
                }), (module, targets)


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


def test_bayesian_and_experimental_consumers_share_the_canonical_irf_resolver():
    selection = importlib.import_module(f"{IRF}.selection")
    resolver = selection._resolve_reconvolution_irf
    assert resolver.__module__ == f"{IRF}.selection"
    for module in (f"{PACKAGE}.bayesian", f"{MEASUREMENT}.adapters"):
        assert importlib.import_module(module)._resolve_reconvolution_irf is resolver


def test_no_eager_internal_cycles(package_imports):
    records, packages = package_imports
    # The Bayesian -> sampling import is deliberately deferred; the reverse
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
    names = (["estimation.api", "estimator_api", "evaluation_results", "evaluation_core", "evaluation_uncertainty"]
             if generic_only else [*OLD_MODULE_PATHS, "estimation.api", "estimation.baselines",
                                   "physics.models", "physics.convolution", "physics.forward_model",
                                   "preprocessing.histograms", "preprocessing.features",
                                   "preprocessing.representations",
                                   "irf.profiles", "irf.preparation", "irf.estimation", "irf.selection",
                                   "measurement.records", "measurement.io", "measurement.adapters"])
    if reverse:
        names.reverse()
    blocked = ["emcee"]
    if generic_only:
        blocked += [PERSISTENCE, f"{PACKAGE}.bayesian_sampling",
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


@pytest.mark.parametrize("module", ["api", "baselines"])
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
assert 'tcspc_toolkit.estimator_api' not in sys.modules
assert 'tcspc_toolkit.baselines' not in sys.modules
assert 'tcspc_toolkit.persistence' not in sys.modules
assert 'emcee' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []
