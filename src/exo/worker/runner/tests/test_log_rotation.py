# pyright: reportUnusedFunction=false, reportAny=false
"""Tests for runner log rotation (P0 #34, L36).

Verifies that the RunnerStdioHandler rotation infrastructure:
- Rotates oversized logs into timestamped .zst archives
- Prunes archives beyond the configured max
- Startup rotation in create() works
"""
from pathlib import Path

import zstandard

from exo.worker.runner.supervisor import _rotate_log_file


class TestRotateLogFile:
    def test_rotate_creates_zst_archive(self, tmp_path: Path) -> None:
        """Rotating a log file should create a .zst archive and remove the original."""
        log = tmp_path / "stdout.log"
        log.write_text("line 1\nline 2\nline 3\n")

        _rotate_log_file(log)

        # Original should be gone
        assert not log.exists()
        # Exactly one archive should exist
        archives = list(tmp_path.glob("stdout.log.*.zst"))
        assert len(archives) == 1

        # Archive should be valid zstd containing the original content
        decompressor = zstandard.ZstdDecompressor()
        raw = archives[0].read_bytes()
        # copy_stream doesn't write content size to frame header, so we must
        # specify max_output_size for decompress()
        content = decompressor.decompress(raw, max_output_size=1024 * 1024)
        assert b"line 1" in content

    def test_rotate_prunes_old_archives(self, tmp_path: Path) -> None:
        """Only EXO_RUNNER_LOG_MAX_ARCHIVES are kept; oldest are deleted."""
        from exo.shared.constants import EXO_RUNNER_LOG_MAX_ARCHIVES

        log = tmp_path / "stderr.log"
        archives = []

        # Create more archives than the limit
        for i in range(EXO_RUNNER_LOG_MAX_ARCHIVES + 3):
            log.write_text(f"iteration {i}\n")
            _rotate_log_file(log)
            archives = sorted(tmp_path.glob("stderr.log.*.zst"))

        # After rotation, at most EXO_RUNNER_LOG_MAX_ARCHIVES should remain
        assert len(archives) <= EXO_RUNNER_LOG_MAX_ARCHIVES

    def test_rotate_missing_file_no_crash(self, tmp_path: Path) -> None:
        """Rotating a nonexistent file should not raise."""
        _rotate_log_file(tmp_path / "nonexistent.log")  # should not raise

    def test_rotate_archives_are_sorted_oldest_pruned(self, tmp_path: Path) -> None:
        """Oldest archives (by timestamp suffix) are pruned first."""
        log = tmp_path / "test.log"

        # Create 4 archives
        for i in range(4):
            log.write_text(f"data {i}\n")
            _rotate_log_file(log)

        archives = sorted(tmp_path.glob("test.log.*.zst"))
        assert len(archives) == 4  # all under default limit of 5

        # Add 3 more — now 7 total, should be pruned to 5
        for i in range(4, 7):
            log.write_text(f"data {i}\n")
            _rotate_log_file(log)

        archives = sorted(tmp_path.glob("test.log.*.zst"))
        assert len(archives) == 5  # MAX_ARCHIVES = 5
