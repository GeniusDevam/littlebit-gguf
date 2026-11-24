import hashlib
import re
import subprocess
from pathlib import Path

import pytest

from convert_littlebit_to_gguf import LLAMA, ROOT

SCRIPT = ROOT / "scripts" / "apply-patches.sh"
PATCHES = sorted((ROOT / "patches").glob("*.diff"))
# Patch 0001 must stay as it was in revision 3 so that trees patched by that revision upgrade.
PATCH_1_SHA256 = "6f6c863df237d028ff8c9e3da65f227bcfe4d93b7936b32e456e6a6ce5a74d03"


def touched() -> list[str]:
    text = "".join(patch.read_text(encoding="utf-8") for patch in PATCHES)
    return sorted(set(re.findall(r"^diff --git a/(\S+) b/", text, re.M)))


def snapshot(tree: Path) -> dict[str, bytes]:
    return {name: (tree / name).read_bytes() for name in touched()}


def git_apply(tree: Path, patch: Path) -> None:
    subprocess.run(["git", "apply", str(patch)], cwd=tree, check=True)


def run_script(tree: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), str(tree)], capture_output=True, text=True, check=False
    )


@pytest.fixture
def clean_tree(tmp_path: Path) -> Path:
    if not (LLAMA / "src").is_dir():
        pytest.skip("llama.cpp submodule is not checked out")
    tree = tmp_path / "llama.cpp"
    tree.mkdir()
    archive = subprocess.run(
        ["git", "-C", str(LLAMA), "archive", "HEAD", *touched()],
        capture_output=True,
        check=True,
    )
    subprocess.run(["tar", "-x", "-C", str(tree)], input=archive.stdout, check=True)
    return tree


@pytest.fixture
def patched_tree(tmp_path: Path, clean_tree: Path) -> dict[str, bytes]:
    expected = tmp_path / "expected"
    expected.mkdir()
    subprocess.run(["cp", "-r", f"{clean_tree}/.", str(expected)], check=True)
    for patch in PATCHES:
        git_apply(expected, patch)
    return snapshot(expected)


def test_first_patch_is_unchanged() -> None:
    digest = hashlib.sha256(PATCHES[0].read_bytes()).hexdigest()
    assert PATCHES[0].name == "0001-littlebit.diff"
    assert digest == PATCH_1_SHA256


@pytest.mark.parametrize("already", range(len(PATCHES)))
def test_applies_what_is_missing(
    clean_tree: Path, patched_tree: dict[str, bytes], already: int
) -> None:
    # already = 0 is a clean tree, 1 is a tree patched by revision 3, and so on
    for patch in PATCHES[:already]:
        git_apply(clean_tree, patch)
    result = run_script(clean_tree)
    assert result.returncode == 0, result.stderr
    assert snapshot(clean_tree) == patched_tree


def test_second_run_changes_nothing(
    clean_tree: Path, patched_tree: dict[str, bytes]
) -> None:
    assert run_script(clean_tree).returncode == 0
    result = run_script(clean_tree)
    assert result.returncode == 0, result.stderr
    assert snapshot(clean_tree) == patched_tree


def test_other_edits_are_kept(clean_tree: Path) -> None:
    for patch in PATCHES:
        git_apply(clean_tree, patch)
    path = clean_tree / "src" / "llama-model.cpp"
    path.write_text(path.read_text(encoding="utf-8") + "// local edit\n")
    before = snapshot(clean_tree)
    result = run_script(clean_tree)
    assert result.returncode == 0, result.stderr
    assert snapshot(clean_tree) == before


@pytest.mark.parametrize("patched", [False, True])
def test_conflicting_edit_stops_and_changes_nothing(
    clean_tree: Path, patched: bool
) -> None:
    if patched:
        for patch in PATCHES:
            git_apply(clean_tree, patch)
    path = clean_tree / "src" / "llama-graph.cpp"
    text = path.read_text(encoding="utf-8")
    marker = "a Hadamard-folded weight reads the transformed input"
    assert marker in text
    path.write_text(text.replace(marker, "a folded weight reads the input"))
    before = snapshot(clean_tree)
    result = run_script(clean_tree)
    assert result.returncode != 0
    assert re.search(r"error: 000\d-\S+\.diff does not apply", result.stderr)
    assert snapshot(clean_tree) == before
