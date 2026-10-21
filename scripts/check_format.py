"""Check Ruff formatting while allowing only unchanged, audited formatting debt.

There is deliberately no baseline-generation/update command. CI compares the
baseline with the PR base (or pre-push commit); local runs compare with HEAD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib


BASELINE = ".ruff-format-baseline.json"
SCOPES = ("src", "tests", "examples", "scripts")


def normalized_hash(data: bytes) -> str:
    """Ignore checkout line-ending conversion, but no other source changes."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def baseline_errors(current: dict, previous: dict | None) -> list[str]:
    """Existing exemptions can only be removed, never added or rehashed."""
    errors = []
    if current.get("schema") != 1 or not isinstance(current.get("files"), dict):
        return ["Invalid formatting baseline schema"]
    for name, digest in current["files"].items():
        path = Path(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or path.parts[0] not in SCOPES
            or path.suffix != ".py"
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            errors.append(f"Invalid baseline entry: {name}")
    if previous is not None:
        if current.get("ruff_version") != previous.get("ruff_version"):
            errors.append("Ruff migration requires an explicit policy review")
        for name, digest in current["files"].items():
            if previous["files"].get(name) != digest:
                errors.append(f"Added or rewritten formatting exemption: {name}")
    return errors


def formatting_errors(
    unformatted: set[str], hashes: dict[str, str], baseline: dict
) -> list[str]:
    return [
        f"Format changed/new file: {name}"
        for name in sorted(unformatted)
        if hashes.get(name) != baseline["files"].get(name)
        or name not in baseline["files"]
    ]


def previous_baseline(root: Path, base_ref: str) -> dict | None:
    # An invalid/missing Git revision is an error, not a bootstrap exemption.
    subprocess.run(
        ["git", "rev-parse", "--verify", f"{base_ref}^{{commit}}"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    tree = subprocess.run(
        ["git", "ls-tree", "--name-only", base_ref, "--", BASELINE],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if not tree.stdout.strip():
        return None  # One-time introduction, reviewed in Issue #15.
    return json.loads(
        subprocess.run(
            ["git", "show", f"{base_ref}:{BASELINE}"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", default="HEAD")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    current = json.loads((root / BASELINE).read_text(encoding="utf-8"))
    errors = baseline_errors(current, previous_baseline(root, args.base_ref))
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = current["ruff_version"]
    if config["tool"]["ruff"]["required-version"] != f"=={version}":
        errors.append("Ruff config and baseline versions differ")
    if f"ruff=={version}" not in config["project"]["optional-dependencies"]["dev"]:
        errors.append("Development dependency must pin the baseline Ruff version")
    command = subprocess.run(
        [sys.executable, "-m", "ruff", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    if command.stdout.strip() != f"ruff {version}":
        errors.append(f"Install pinned Ruff {version}")
    if errors:
        print("\n".join(errors))
        return 1
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--check",
            "--output-format=json",
            "--no-cache",
            *SCOPES,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode not in (0, 1):
        print(result.stderr, file=sys.stderr)
        return result.returncode
    diagnostics = json.loads(result.stdout)
    unformatted = set()
    for item in diagnostics:
        if item["code"] != "unformatted":
            errors.append(item["message"])
        else:
            unformatted.add(Path(item["filename"]).relative_to(root).as_posix())
    hashes = {name: normalized_hash((root / name).read_bytes()) for name in unformatted}
    errors.extend(formatting_errors(unformatted, hashes, current))
    if errors:
        print("\n".join(errors))
        print("Run: python -m ruff format <listed files>; do not refresh the baseline.")
        return 1
    print(f"Formatting PASS: {len(unformatted)} unchanged audited debt files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
