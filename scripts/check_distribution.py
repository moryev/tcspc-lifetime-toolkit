"""Inspect wheel/sdist and exercise installed artifacts outside the checkout.

Creates disposable base-only virtual environments, then adds the Bayesian extra
for a small integration smoke. No full MCMC example or reference study runs.
"""

from __future__ import annotations

import argparse
import ast
from email.parser import BytesParser
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile
import tomllib
import venv
import zipfile


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def literal(path: Path, name: str):
    """Reuse reviewed contract literals without importing tests or the package."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing contract {name} in {path}")


def package_contract(root: Path) -> dict:
    structure = root / "tests/test_package_structure.py"
    retained = literal(structure, "RETAINED_PACKAGE_APIS")
    files = (
        literal(structure, "CLOSURE_CANONICAL_FILES")
        | literal(structure, "CLOSURE_SUPPORT_FILES")
        | {f"{name}/__init__.py" for name in retained}
    )
    require(len(files) == 61, "Production module inventory changed")
    return {
        "files": sorted(files),
        "removed": list(literal(structure, "REMOVED_FLAT_FACADES")),
        "exports": literal(root / "tests/test_public_api.py", "EXPECTED_ROOT_EXPORTS"),
    }


def check_metadata(raw: bytes, project: dict) -> None:
    metadata = BytesParser().parsebytes(raw)
    require(metadata["Name"] == project["name"], "Distribution name mismatch")
    require(metadata["Version"] == project["version"], "Distribution version mismatch")
    require(metadata["Requires-Python"] == ">=3.11", "Python requirement changed")
    require(metadata["License-Expression"] == "Apache-2.0", "Missing SPDX license")
    require(
        "LICENSE" in metadata.get_all("License-File", []), "Missing license metadata"
    )
    requirements = metadata.get_all("Requires-Dist", [])
    for dependency in project["dependencies"]:
        require(dependency in requirements, f"Missing requirement: {dependency}")
    require(
        any(
            line.startswith("emcee<4,>=3.1.6;") and "bayesian" in line
            for line in requirements
        ),
        "Missing optional Bayesian dependency",
    )
    require(metadata["Author-email"] is not None, "Missing author metadata")


def extract_sdist(archive: Path, destination: Path) -> Path:
    """Accept regular source files/directories only; never follow archive links."""
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        roots = {PurePosixPath(member.name).parts[0] for member in members}
        require(len(roots) == 1, "sdist must have one top-level directory")
        for member in members:
            path = PurePosixPath(member.name)
            require(
                not path.is_absolute()
                and ".." not in path.parts
                and "\\" not in member.name
                and (member.isdir() or member.isfile()),
                f"Unsafe sdist member: {member.name}",
            )
        bundle.extractall(destination, filter="data")
    return destination / roots.pop()


def inspect_artifacts(
    root: Path, wheel: Path, source: Path, project: dict, contract: dict
):
    with zipfile.ZipFile(wheel) as bundle:
        names = bundle.namelist()
        actual = {
            name.removeprefix("tcspc_toolkit/")
            for name in names
            if name.startswith("tcspc_toolkit/") and name.endswith(".py")
        }
        require(actual == set(contract["files"]), "Wheel module inventory differs")
        metadata = [name for name in names if name.endswith(".dist-info/METADATA")]
        require(len(metadata) == 1, "Wheel metadata missing/ambiguous")
        check_metadata(bundle.read(metadata[0]), project)
        require(
            any(name.endswith("/licenses/LICENSE") for name in names),
            "No wheel license",
        )
        entries = next(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        require(
            b"tcspc = tcspc_toolkit.cli:main" in bundle.read(entries),
            "Console entry point missing",
        )
        require(
            all(
                name.startswith("tcspc_toolkit/") or ".dist-info/" in name
                for name in names
            ),
            "Unexpected top-level wheel content",
        )
    check_metadata((source / "PKG-INFO").read_bytes(), project)
    # All tracked redistributable workflows must be included, byte-for-byte
    # modulo Git's checkout line endings. Ignore generated and local materials.
    tracked = (
        subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
        )
        .stdout.decode()
        .split("\0")
    )
    prefixes = (
        "src/tcspc_toolkit/",
        "tests/",
        "examples/",
        "scripts/",
        "configs/",
        "docs/",
        "notebooks/",
        "data/examples/",
        "data/reference/",
    )
    required = {
        ".github/workflows/ci.yml",
        ".gitignore",
        "LICENSE",
        "CITATION.cff",
        "CHANGELOG.md",
        "README.md",
        "pyproject.toml",
        "AGENTS.md",
        "MANIFEST.in",
        ".ruff-format-baseline.json",
    }
    # A reviewed deletion in an uncommitted release-preparation diff should
    # disappear from the sdist; only files present in this checkout can ship.
    required.update(
        name
        for name in tracked
        if name.startswith(prefixes) and (root / name).is_file()
    )
    required.add("tests/fixtures/week7_sample_770_windows_reference.json")
    # Include new, uncommitted Issue-15 tooling too.
    required.update(
        path.relative_to(root).as_posix()
        for folder in ("scripts", "tests")
        for path in (root / folder).glob("*.py")
    )
    for name in sorted(required):
        require((source / name).is_file(), f"sdist missing {name}")
        require(
            (source / name).read_bytes().replace(b"\r\n", b"\n")
            == (root / name).read_bytes().replace(b"\r\n", b"\n"),
            f"sdist content differs: {name}",
        )
    require(not (source / "data/generated").exists(), "Generated science in sdist")
    require(not (source / ".git").exists(), "Git internals in sdist")


def run(command: list[str], cwd: Path) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["MPLBACKEND"] = "Agg"
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=cwd, env=environment, check=True, timeout=600)


def installed_smokes(
    python: Path, source: Path, work: Path, contract: dict, version: str
):
    probe = """
import importlib, importlib.metadata, importlib.util, pathlib, sys
import tcspc_toolkit
contract = CONTRACT
assert importlib.metadata.version("tcspc-lifetime-toolkit") == VERSION
assert pathlib.Path(tcspc_toolkit.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix))
assert tuple(tcspc_toolkit.__all__) == tuple(contract["exports"])
assert len(contract["exports"]) == 66 and len(contract["removed"]) == 38
assert importlib.util.find_spec("emcee") is None
for name in contract["removed"]:
    assert importlib.util.find_spec("tcspc_toolkit." + name) is None
for name in contract["files"]:
    module = "tcspc_toolkit." + name.removesuffix(".py").replace("/", ".")
    importlib.import_module(module.removesuffix(".__init__"))
print("Installed canonical imports, 66 exports, 38 removals and base-only optional isolation PASS")
"""
    run(
        [
            str(python),
            "-I",
            "-c",
            probe.replace("CONTRACT", repr(contract)).replace("VERSION", repr(version)),
        ],
        work,
    )
    readme = (source / "README.md").read_text(encoding="utf-8")
    section = readme.split("<!-- quickstart begin -->", 1)[1].split(
        "<!-- quickstart end -->", 1
    )[0]
    code = re.search(r"```python\n(.*?)```", section, re.S).group(1)
    run([str(python), "-I", "-c", code], work)
    cli = python.parent / ("tcspc.exe" if os.name == "nt" else "tcspc")
    run([str(cli), "--help"], work)
    run(
        [
            str(cli),
            "simulate",
            "--bins",
            "512",
            "--lifetime",
            "2.5",
            "--random-seed",
            "42",
        ],
        work,
    )
    run(
        [
            str(cli),
            "fit",
            "--input",
            str(source / "data/examples/ideal_tcspc_decay.csv"),
            "--time-unit",
            "ns",
            "--data-kind",
            "raw_counts",
        ],
        work,
    )
    for name in (
        "quickstart",
        "experimental_csv",
        "preprocessing_features",
        "ml_lifetime_prediction",
        "uncertainty_example",
        "custom_estimator",
        "persistence_roundtrip",
    ):
        args = (
            ["--output", str(work / "results.sqlite")]
            if name == "persistence_roundtrip"
            else []
        )
        run([str(python), "-I", str(source / "examples" / f"{name}.py"), *args], work)
    run(
        [
            str(python),
            "-I",
            str(source / "scripts/run_issue12_integration.py"),
            "--block-emcee",
        ],
        work,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    distribution = args.dist_dir.resolve()
    wheels, sdists = (
        list(distribution.glob("*.whl")),
        list(distribution.glob("*.tar.gz")),
    )
    require(
        len(wheels) == len(sdists) == 1,
        "Use a directory with exactly one wheel and one sdist",
    )
    contract = package_contract(root)
    with tempfile.TemporaryDirectory(prefix="tcspc-distribution-") as temporary:
        directory = Path(temporary)
        source = extract_sdist(sdists[0], directory / "source")
        inspect_artifacts(root, wheels[0], source, project, contract)
        for kind, artifact in (("wheel", wheels[0]), ("sdist", sdists[0])):
            environment = directory / f"{kind}-env"
            venv.EnvBuilder(with_pip=True).create(environment)
            python = environment / (
                "Scripts/python.exe" if os.name == "nt" else "bin/python"
            )
            work = directory / f"{kind}-work"
            work.mkdir()
            run([str(python), "-m", "pip", "install", str(artifact)], work)
            run([str(python), "-m", "pip", "check"], work)
            installed_smokes(python, source, work, contract, project["version"])
            run([str(python), "-m", "pip", "install", f"{artifact}[bayesian]"], work)
            run([str(python), "-m", "pip", "check"], work)
            run(
                [
                    str(python),
                    "-I",
                    str(source / "scripts/run_issue12_integration.py"),
                    "--bayesian",
                ],
                work,
            )
            run([str(python), "-m", "pip", "freeze"], work)
    print(
        json.dumps(
            {
                "status": "PASS",
                "version": project["version"],
                "artifacts": [path.name for path in (wheels[0], sdists[0])],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
