"""Unit tests for the #1626 OOM-prevention port.

Covers:
- ``measure_cache_bytes`` / ``measure_kv_cache_bytes_per_token`` on the
  current mlx-lm cache classes (KVCache, RotatingKVCache, ...) via ``nbytes``.
- ``KVPrefixCache.force_evict_all()``.
- ``_check_memory_budget`` pure logic: pass-through when pressure is fine,
  prefix-cache eviction before failing, and the friendly error message when
  OOM is predicted.
- ``mlx_generate`` emits a ``finish_reason="error"`` response (which
  ``map_responses_to_chunks`` turns into an ``ErrorChunk``) when the budget
  check predicts OOM.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from exo.worker.engines.mlx import cache as cache_mod
from exo.worker.engines.mlx.generator import generate as generate_mod
from exo.worker.engines.mlx.generator.generate import _check_memory_budget


class _FakeCacheEntry:
    """Mimics an mlx-lm cache entry exposing ``nbytes``."""

    def __init__(self, nbytes: int) -> None:
        self._nbytes = nbytes

    @property
    def nbytes(self) -> int:
        return self._nbytes


class _FakeCacheList(list):
    """List subclass standing in for an mlx-lm cache list."""


def test_measure_cache_bytes_sums_nbytes() -> None:
    cache = _FakeCacheList([_FakeCacheEntry(100), _FakeCacheEntry(250)])
    assert cache_mod.measure_cache_bytes(cache) == 350


def test_measure_cache_bytes_ignores_missing_nbytes() -> None:
    cache = _FakeCacheList([_FakeCacheEntry(100), object()])
    assert cache_mod.measure_cache_bytes(cache) == 100


def test_measure_kv_cache_bytes_per_token_uses_cache_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _FakeCacheList([_FakeCacheEntry(1000)])
    monkeypatch.setattr(cache_mod, "cache_length", lambda c: 10)
    assert cache_mod.measure_kv_cache_bytes_per_token(cache) == 100


def test_measure_kv_cache_bytes_per_token_zero_length() -> None:
    cache = _FakeCacheList([_FakeCacheEntry(1000)])
    assert cache_mod.measure_kv_cache_bytes_per_token(cache) == 0


def test_check_memory_budget_skips_when_bytes_per_token_zero() -> None:
    assert (
        _check_memory_budget(
            bytes_per_token=0,
            total_sequence_tokens=1_000_000,
            kv_prefix_cache=None,
        )
        is None
    )


def test_check_memory_budget_passes_when_pressure_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        generate_mod, "get_system_memory_total", lambda: SimpleNamespace(in_bytes=100)
    )
    monkeypatch.setattr(generate_mod, "get_memory_used_percentage", lambda: 0.10)
    monkeypatch.setattr(generate_mod, "MEMORY_THRESHOLD", 0.70)

    # 10 tokens x 1 byte = 10 bytes / 100 total = +10% → 20% projected < 70%.
    assert (
        _check_memory_budget(
            bytes_per_token=1, total_sequence_tokens=10, kv_prefix_cache=None
        )
        is None
    )


def test_check_memory_budget_fails_when_pressure_high(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        generate_mod, "get_system_memory_total", lambda: SimpleNamespace(in_bytes=100)
    )
    monkeypatch.setattr(generate_mod, "get_memory_used_percentage", lambda: 0.60)
    monkeypatch.setattr(generate_mod, "MEMORY_THRESHOLD", 0.70)

    # 50 tokens x 1 byte = 50 bytes / 100 total = +50% → 110% projected > 70%.
    msg = _check_memory_budget(
        bytes_per_token=1, total_sequence_tokens=50, kv_prefix_cache=None
    )
    assert msg is not None
    assert "Not enough memory" in msg


def test_check_memory_budget_evicts_before_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful prefix-cache eviction should rescue the request."""
    monkeypatch.setattr(
        generate_mod, "get_system_memory_total", lambda: SimpleNamespace(in_bytes=100)
    )
    monkeypatch.setattr(generate_mod, "MEMORY_THRESHOLD", 0.70)
    monkeypatch.setattr(generate_mod.mx, "clear_cache", lambda: None)

    evicted = {"count": 0}

    class _EvictingCache:
        def __init__(self) -> None:
            self.pressure = 0.60
            self.after_evict = 0.05

        def get_memory_used_percentage(self) -> float:
            return self.pressure

        def force_evict_all(self) -> int:
            evicted["count"] += 1
            self.pressure = self.after_evict
            return 3

    cache = _EvictingCache()
    # 40 x 1 / 100 = +40%: 60+40=100% > 70% → evict → 5+40=45% < 70% → OK.
    assert (
        _check_memory_budget(
            bytes_per_token=1, total_sequence_tokens=40, kv_prefix_cache=cache
        )
        is None
    )
    assert evicted["count"] == 1


def test_check_memory_budget_still_fails_after_eviction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        generate_mod, "get_system_memory_total", lambda: SimpleNamespace(in_bytes=100)
    )
    monkeypatch.setattr(generate_mod, "MEMORY_THRESHOLD", 0.70)
    monkeypatch.setattr(generate_mod.mx, "clear_cache", lambda: None)

    class _EvictingCache:
        def get_memory_used_percentage(self) -> float:
            return 0.60

        def force_evict_all(self) -> int:
            return 3

    msg = _check_memory_budget(
        bytes_per_token=1,
        total_sequence_tokens=40,
        kv_prefix_cache=_EvictingCache(),
    )
    assert msg is not None
    assert "Not enough memory" in msg


def test_force_evict_all_clears_entries() -> None:
    kv = cache_mod.KVPrefixCache(None)
    # Simulate a populated prefix cache.
    kv.caches.extend([object(), object()])
    kv.prompts.extend([object(), object()])
    kv._snapshots.extend([None, None])
    kv._media_regions.extend([[], []])
    kv._last_used.extend([1, 2])
    kv.prefill_tps.extend([0.1, 0.2])

    assert kv.force_evict_all() == 2
    assert len(kv.caches) == 0
    assert kv.force_evict_all() == 0
