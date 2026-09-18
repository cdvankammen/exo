"""Tests for bounded size-based log rotation (FIX t_572b98fd).

Regression: commit d16131ee introduced ``_size_rotation`` but never wired it
into ``logger.add`` — the sink still passed ``rotation=lambda _, __:
next(rotate_once)`` referencing the removed ``_once_then_never()`` generator.
Result: exo.log kept its old one-shot rotation and grew unbounded (14GB
observed on a Linux node; 1.77GB on mini1) until the disk filled and
OSError 28 killed the process.

These tests pin the predicate *and* the full sink wiring so neither can rot
again.
"""

from pathlib import Path
from unittest.mock import patch

from loguru import logger

from exo.shared import logging as L


def test_size_rotation_predicate_small_file_no_rotate(tmp_path: Path) -> None:
    log_file = tmp_path / "exo.log"
    log_file.write_bytes(b"x" * 100)
    assert L._size_rotation(log_file, None, None) is False


def test_size_rotation_predicate_oversize_file_rotates(tmp_path: Path) -> None:
    log_file = tmp_path / "exo.log"
    log_file.write_bytes(b"x" * (L._LOG_ROTATION_BYTES + 1))
    assert L._size_rotation(log_file, None, None) is True


def test_size_rotation_predicate_missing_file_no_crash(tmp_path: Path) -> None:
    log_file = tmp_path / "missing.log"
    assert L._size_rotation(log_file, None, None) is False


def test_logger_setup_rotates_and_compresses(tmp_path: Path) -> None:
    """End-to-end: writing past the threshold rotates + zstd-compresses and
    retires the archive; the active file stays bounded and retention caps at
    _MAX_LOG_ARCHIVES."""
    log_file = tmp_path / "exo.log"
    old = L._LOG_ROTATION_BYTES
    L._LOG_ROTATION_BYTES = 1024  # shrink so the test is fast
    try:
        L.logger_setup(log_file, verbosity=1)
        for i in range(2000):
            logger.info(f"spam {i} " + "y" * 40)
        logger.complete()
    finally:
        L._LOG_ROTATION_BYTES = old

    archives = sorted(tmp_path.glob("*.log.zst"))
    assert len(archives) >= 1, "expected at least one rotated archive"
    assert len(archives) <= L._MAX_LOG_ARCHIVES, "retention cap exceeded"
    assert log_file.stat().st_size <= 5000, "active log grew unbounded"