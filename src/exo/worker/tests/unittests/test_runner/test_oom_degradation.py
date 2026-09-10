# type: ignore
"""OOM graceful degradation tests (P1, #2182/#1872)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from exo.shared.types.worker.runners import RunnerDegraded
from exo.worker.runner.diagnostics import (
    RunnerOutOfMemory,
    _parse_oom_error,
)
from exo.worker.runner.runner import (
    OOM_MIN_CONCURRENCY,
    OOM_RETRY_LIMIT,
    Runner,
)


# --- _is_oom detection ---

class FakeRunnerForOomDetection:
    _is_oom = Runner._is_oom

    def __init__(self):
        self._oom_recovery_count = 0
        self._oom_original_max_concurrent = -1


def test_is_oom_detects_failed_to_allocate():
    r = FakeRunnerForOomDetection()
    assert r._is_oom(RuntimeError("Failed to allocate 1024 bytes"))


def test_is_oom_detects_insufficient_memory():
    r = FakeRunnerForOomDetection()
    assert r._is_oom(RuntimeError("[METAL] Insufficient memory to allocate"))


def test_is_oom_detects_bad_alloc():
    r = FakeRunnerForOomDetection()
    assert r._is_oom(RuntimeError("std::bad_alloc"))


def test_is_oom_detects_out_of_memory():
    r = FakeRunnerForOomDetection()
    assert r._is_oom(Exception("out of memory"))


def test_is_oom_rejects_non_oom():
    r = FakeRunnerForOomDetection()
    assert not r._is_oom(RuntimeError("CUDA timeout"))
    assert not r._is_oom(ValueError("shape mismatch"))
    assert not r._is_oom(KeyError("missing"))


# --- _handle_oom recovery ---

def test_handle_oom_clears_memory_and_halves_concurrency():
    fake_gen = MagicMock()
    fake_gen._max_batch_size = 8
    fake_self = SimpleNamespace(
        generator=fake_gen,
        _oom_recovery_count=0,
        _oom_original_max_concurrent=-1,
        update_status=MagicMock(),
        event_sender=MagicMock(),
        runner_id="test",
    )
    exc = RuntimeError("Failed to allocate")
    result = Runner._handle_oom(fake_self, exc)
    assert result is True
    assert fake_self._oom_recovery_count == 1
    assert fake_gen.clear_memory.called
    assert fake_gen._max_batch_size == 4
    assert fake_self._oom_original_max_concurrent == 8
    # Check RunnerDegraded was emitted
    fake_self.update_status.assert_called_once()
    degraded = fake_self.update_status.call_args[0][0]
    assert isinstance(degraded, RunnerDegraded)
    assert degraded.original_concurrency == 8
    assert degraded.current_concurrency == 4


def test_handle_oom_exhausts_retry_budget():
    fake_gen = MagicMock()
    fake_gen._max_batch_size = 1
    fake_self = SimpleNamespace(
        generator=fake_gen,
        _oom_recovery_count=OOM_RETRY_LIMIT,
        _oom_original_max_concurrent=1,
        update_status=MagicMock(),
        event_sender=MagicMock(),
        runner_id="test",
    )
    result = Runner._handle_oom(fake_self, RuntimeError("OOM"))
    assert result is False


def test_handle_oom_never_below_min_concurrency():
    fake_gen = MagicMock()
    fake_gen._max_batch_size = 2
    fake_self = SimpleNamespace(
        generator=fake_gen,
        _oom_recovery_count=0,
        _oom_original_max_concurrent=-1,
        update_status=MagicMock(),
        event_sender=MagicMock(),
        runner_id="test",
    )
    Runner._handle_oom(fake_self, RuntimeError("OOM"))
    assert fake_gen._max_batch_size >= OOM_MIN_CONCURRENCY


# --- RunnerDegraded status ---

def test_runner_degraded_fields():
    d = RunnerDegraded(
        reason="OOM test",
        original_concurrency=8,
        current_concurrency=4,
        prefill_server_port=12345,
    )
    assert d.original_concurrency == 8
    assert d.current_concurrency == 4
    assert d.prefill_server_port == 12345
    assert "OOM" in d.reason


def test_runner_degraded_defaults():
    d = RunnerDegraded()
    assert d.reason == ""
    assert d.original_concurrency == 0
    assert d.current_concurrency == 0
    assert d.prefill_server_port is None


# --- diagnostics: _parse_oom_error ---

def test_parse_oom_error_detects_metal_alloc():
    line = "libc++abi: terminating with uncaught exception of type std::runtime_error: [METAL] Failed to allocate 2097152 bytes"
    diag = _parse_oom_error(line, (line,))
    assert diag is not None
    assert isinstance(diag, RunnerOutOfMemory)
    assert diag.peak_memory_bytes == 2097152
    assert "Failed to allocate" in diag.message


def test_parse_oom_error_no_match():
    assert _parse_oom_error("some other error", ()) is None
    assert _parse_oom_error("GPU timeout", ()) is None
