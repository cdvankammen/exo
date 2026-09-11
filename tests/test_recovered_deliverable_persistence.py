# type: ignore
"""Persistence tests for recovered deliverable t_97ebac23.

Asserts that the recovered exo-dashboard-automation deliverable files exist at
the canonical persistent path and survive workspace cleanup. These tests are
independent of exo runtime dependencies (stdlib + pytest only) so they can run
in any environment without cluster or model prerequisites.

Recovery target (defined by audit task t_a642e9c6):
    /Users/chris/Documents/hermesResearch/exo-research/V2/notes/exo-dashboard-automation/

Expected files:
    scripts/exo_api.py      - FastAPI HTTP client for exo dashboard
    scripts/exo_browser.py  - PinchTab browser automation script
    scripts/verify.py       - Structural verification script
    README.md               - Usage documentation

Run with:
    pytest tests/test_recovered_deliverable_persistence.py -v
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

# Canonical persistent path defined by audit task t_a642e9c6
RECOVERY_TARGET = Path(
    "/Users/chris/Documents/hermesResearch/exo-research/V2/notes/exo-dashboard-automation"
)

# The 4 files that constitute the recovered deliverable
EXPECTED_FILES = [
    "scripts/exo_api.py",
    "scripts/exo_browser.py",
    "scripts/verify.py",
    "README.md",
]


class TestRecoveredDeliverableExists:
    """Assert each expected file exists at the persistent recovery path."""

    @pytest.mark.parametrize("rel_path", EXPECTED_FILES)
    def test_file_exists(self, rel_path: str) -> None:
        full_path = RECOVERY_TARGET / rel_path
        assert full_path.exists(), (
            f"Recovered deliverable file missing: {full_path}\n"
            f"Expected at persistent path: {RECOVERY_TARGET}"
        )

    @pytest.mark.parametrize("rel_path", EXPECTED_FILES)
    def test_file_non_empty(self, rel_path: str) -> None:
        full_path = RECOVERY_TARGET / rel_path
        if not full_path.exists():
            pytest.skip(f"File not yet present: {full_path}")
        size = full_path.stat().st_size
        assert size > 0, f"Recovered file is empty: {full_path}"

    def test_recovery_directory_exists(self) -> None:
        assert RECOVERY_TARGET.is_dir(), (
            f"Recovery target directory does not exist: {RECOVERY_TARGET}"
        )

    def test_scripts_subdirectory_exists(self) -> None:
        scripts_dir = RECOVERY_TARGET / "scripts"
        assert scripts_dir.is_dir(), (
            f"Scripts subdirectory missing: {scripts_dir}"
        )


class TestRecoveredDeliverablePersistence:
    """Assert files survive operations that simulate post-task cleanup.

    The original loss (t_97ebac23) was caused by scratch-only delivery with
    workspace GC. These tests verify the recovered files are on persistent
    storage and NOT inside any kanban scratch workspace.
    """

    def test_not_in_scratch_workspace(self) -> None:
        """Files must NOT reside under any kanban scratch workspace path."""
        scratch_marker = "/.hermes/kanban/boards/"
        resolved = RECOVERY_TARGET.resolve()
        assert scratch_marker not in str(resolved), (
            f"Recovery path is inside a kanban workspace (will be GC'd): {resolved}"
        )

    def test_not_in_tmp(self) -> None:
        """Files must NOT reside under /tmp or system temp directories."""
        resolved = str(RECOVERY_TARGET.resolve())
        tmp_prefixes = ["/tmp/", "/var/folders/", tempfile.gettempdir()]
        for prefix in tmp_prefixes:
            assert not resolved.startswith(prefix), (
                f"Recovery path is under temporary directory: {resolved}"
            )

    def test_files_survive_copy_roundtrip(self) -> None:
        """Simulate persistence: copy all files to a temp dir, verify integrity.

        This models the post-cleanup scenario where only the persistent path
        remains. If files can be faithfully copied and verified, they are
        real durable files (not symlinks to scratch or ephemeral mounts).
        """
        if not RECOVERY_TARGET.is_dir():
            pytest.skip(f"Recovery directory not yet present: {RECOVERY_TARGET}")

        with tempfile.TemporaryDirectory(prefix="exo_recovery_verify_") as tmpdir:
            dest = Path(tmpdir) / "recovered"
            shutil.copytree(RECOVERY_TARGET, dest)

            for rel_path in EXPECTED_FILES:
                copied = dest / rel_path
                original = RECOVERY_TARGET / rel_path
                if not original.exists():
                    continue
                assert copied.exists(), f"Copy failed for: {rel_path}"
                assert copied.stat().st_size == original.stat().st_size, (
                    f"Size mismatch after copy: {rel_path}"
                )
                # Verify content matches byte-for-byte
                assert copied.read_bytes() == original.read_bytes(), (
                    f"Content mismatch after copy: {rel_path}"
                )

    def test_files_are_regular_files(self) -> None:
        """Recovered files must be regular files, not symlinks to scratch."""
        for rel_path in EXPECTED_FILES:
            full_path = RECOVERY_TARGET / rel_path
            if not full_path.exists():
                continue
            assert not full_path.is_symlink(), (
                f"File is a symlink (may point to GC'd scratch): {full_path} -> {full_path.resolve()}"
            )
            assert full_path.is_file(), (
                f"Not a regular file: {full_path}"
            )


class TestRecoveredDeliverableContent:
    """Basic content sanity checks for recovered files."""

    def test_python_files_have_valid_syntax(self) -> None:
        """All .py files must compile without syntax errors."""
        import py_compile

        py_files = [f for f in EXPECTED_FILES if f.endswith(".py")]
        for rel_path in py_files:
            full_path = RECOVERY_TARGET / rel_path
            if not full_path.exists():
                continue
            try:
                py_compile.compile(str(full_path), doraise=True)
            except py_compile.PyCompileError as e:
                pytest.fail(f"Syntax error in {rel_path}: {e}")

    def test_readme_has_content(self) -> None:
        """README.md should contain meaningful documentation."""
        readme = RECOVERY_TARGET / "README.md"
        if not readme.exists():
            pytest.skip("README.md not yet present")
        content = readme.read_text()
        # Should have more than just a title
        assert len(content) > 100, "README.md is too short to be useful documentation"

    def test_api_script_references_http(self) -> None:
        """exo_api.py should reference HTTP/API concepts."""
        api_file = RECOVERY_TARGET / "scripts/exo_api.py"
        if not api_file.exists():
            pytest.skip("exo_api.py not yet present")
        content = api_file.read_text().lower()
        has_http_ref = any(
            kw in content for kw in ["http", "request", "api", "client", "endpoint"]
        )
        assert has_http_ref, "exo_api.py doesn't appear to contain HTTP/API client code"
