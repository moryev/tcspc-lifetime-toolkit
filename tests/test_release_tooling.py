"""Regression checks for the formatting ratchet and artifact safety boundaries."""

import copy
import io
from pathlib import Path
import runpy
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def formatter():
    return runpy.run_path(str(ROOT / "scripts/check_format.py"))


def test_line_ending_conversion_is_not_a_source_change(formatter):
    digest = formatter["normalized_hash"]
    assert digest(b"x = 1\n") == digest(b"x = 1\r\n")
    assert digest(b"x = 1\n") != digest(b"x = 2\n")


@pytest.mark.parametrize("change", ["new", "modified", "deleted_then_reintroduced"])
def test_changed_or_new_unformatted_source_fails(formatter, change):
    baseline = {"files": {"src/old.py": "old"}}
    name = "src/new.py" if change == "new" else "src/old.py"
    if change == "deleted_then_reintroduced":
        baseline["files"].clear()
    errors = formatter["formatting_errors"]({name}, {name: "new"}, baseline)
    assert len(errors) == 1 and name in errors[0]


def test_unchanged_debt_and_formatted_edits_pass(formatter):
    baseline = {"files": {"src/old.py": "old"}}
    assert (
        formatter["formatting_errors"]({"src/old.py"}, {"src/old.py": "old"}, baseline)
        == []
    )
    assert formatter["formatting_errors"](set(), {"src/old.py": "new"}, baseline) == []


@pytest.mark.parametrize("change", ["add", "rehash", "version"])
def test_baseline_cannot_be_refreshed_to_bypass_failures(formatter, change):
    previous = {
        "schema": 1,
        "ruff_version": "0.17.1",
        "files": {"src/old.py": "a" * 64},
    }
    current = copy.deepcopy(previous)
    if change == "add":
        current["files"]["src/new.py"] = "b" * 64
    elif change == "rehash":
        current["files"]["src/old.py"] = "b" * 64
    else:
        current["ruff_version"] = "999"
    assert formatter["baseline_errors"](current, previous)


def test_baseline_can_only_shrink_after_bootstrap(formatter):
    previous = {
        "schema": 1,
        "ruff_version": "0.17.1",
        "files": {"src/old.py": "a" * 64},
    }
    current = {**previous, "files": {}}
    assert formatter["baseline_errors"](current, previous) == []
    assert formatter["baseline_errors"](previous, None) == []


@pytest.mark.parametrize(
    "name", ["../outside.py", "/outside.py", "src\\old.py", "data/test.py"]
)
def test_baseline_rejects_paths_outside_policy(formatter, name):
    assert formatter["baseline_errors"](
        {"schema": 1, "ruff_version": "0.17.1", "files": {name: "a" * 64}}, None
    )


def test_invalid_git_ref_is_not_treated_as_first_baseline(formatter):
    import subprocess

    with pytest.raises(subprocess.CalledProcessError):
        formatter["previous_baseline"](ROOT, "refs/heads/issue15-nonexistent-test-ref")


@pytest.mark.parametrize("kind", ["traversal", "symlink"])
def test_sdist_extraction_rejects_unsafe_members(tmp_path, kind):
    checker = runpy.run_path(str(ROOT / "scripts/check_distribution.py"))
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        item = tarfile.TarInfo(
            "package/../../escape" if kind == "traversal" else "package/link"
        )
        if kind == "symlink":
            item.type = tarfile.SYMTYPE
            item.linkname = "../../escape"
            bundle.addfile(item)
        else:
            item.size = 1
            bundle.addfile(item, io.BytesIO(b"x"))
    with pytest.raises(AssertionError, match="Unsafe"):
        checker["extract_sdist"](archive, tmp_path / "out")
    assert not (tmp_path / "escape").exists()


def test_distribution_contract_reuses_reviewed_inventory():
    checker = runpy.run_path(str(ROOT / "scripts/check_distribution.py"))
    contract = checker["package_contract"](ROOT)
    assert len(contract["files"]) == 61
    assert len(contract["exports"]) == 66
    assert len(contract["removed"]) == 38
