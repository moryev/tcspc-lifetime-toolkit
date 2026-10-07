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
# Stage 1 will update the extension set to estimation.api, without removing
# estimator_api from OLD_MODULE_PATHS or relaxing the dependency constraints.
EXTENSION = {f"{PACKAGE}.estimator_api"}
FACTUAL = {f"{PACKAGE}.{name}" for name in (
    "evaluation_results", "evaluation_core", "evaluation_uncertainty",
)}
PHYSICAL = {f"{PACKAGE}.{name}" for name in (
    "models", "convolution", "forward_model", "irf",
)}
PERSISTENCE = f"{PACKAGE}.persistence"


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
    names = (["estimator_api", "evaluation_results", "evaluation_core", "evaluation_uncertainty"]
             if generic_only else list(OLD_MODULE_PATHS))
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
