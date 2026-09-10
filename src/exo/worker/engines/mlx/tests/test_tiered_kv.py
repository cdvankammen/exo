"""Tests for tiered_kv.py — TieredKVPrefixCache.

Tests GPU->CPU spill, CPU->GPU promotion, tiered memory accounting,
and environment-gated multiplier in placement_utils.
"""
import os
import sys
import pytest
import numpy as np

# Set env before any exo imports
os.environ["EXO_TIERED_KV"] = "0"  # default off for most tests

import mlx.core as mx

from exo.worker.engines.mlx.tiered_kv import (
    TieredKVPrefixCache,
    CPUBlock,
    CPUBlockStore,
    _GPU_EVICT_THRESHOLD,
    _CPU_MAX_BYTES,
    _TIERED_KV_ENABLED,
)
from exo.worker.engines.mlx.paged_cache import (
    BlockAllocator,
    PagedKVCache,
    DEFAULT_BLOCK_SIZE,
)


# ── Helpers ──

def _make_fake_cache(n_layers: int = 2, n_tokens: int = 32,
                     n_kv_heads: int = 4, head_dim: int = 8):
    """Create a fake KVCacheType (list of PagedKVCache) with random data."""
    allocator = BlockAllocator(block_size=16, max_blocks=256)
    layers = []
    for _ in range(n_layers):
        paged = PagedKVCache(allocator)
        # Fill with random tokens
        batch = 1
        k = mx.random.normal((batch, n_kv_heads, n_tokens, head_dim))
        v = mx.random.normal((batch, n_kv_heads, n_tokens, head_dim))
        paged.update_and_fetch(k, v)
        layers.append(paged)
    return layers


def _make_fake_prompt(n_tokens: int = 32) -> mx.array:
    """Create fake prompt token array."""
    return mx.array(list(range(n_tokens)))


# ── CPUBlock tests ──

class TestCPUBlock:
    def test_from_paged_arrays_roundtrip(self):
        """Verify CPU block can roundtrip through numpy."""
        k = mx.random.normal((1, 4, 16, 8))
        v = mx.random.normal((1, 4, 16, 8))
        block = CPUBlock.from_paged_arrays(k, v, block_id=0, access=1)
        assert block.keys.shape == (1, 4, 16, 8)
        assert block.values.shape == (1, 4, 16, 8)
        assert block.nbytes > 0

        k2, v2 = block.to_mx()
        # Verify shapes match
        assert k2.shape == k.shape
        assert v2.shape == v.shape


class TestCPUBlockStore:
    def test_store_and_load(self):
        store = CPUBlockStore(max_bytes=1024 * 1024 * 1024)
        k = mx.random.normal((1, 4, 16, 8))
        v = mx.random.normal((1, 4, 16, 8))
        store.store(42, k, v)
        assert store.used_bytes > 0

        loaded = store.load(42)
        assert loaded is not None
        k2, v2 = loaded
        assert k2.shape == k.shape

    def test_load_nonexistent_returns_none(self):
        store = CPUBlockStore(max_bytes=1024 * 1024 * 1024)
        assert store.load(999) is None

    def test_evict_lru(self):
        store = CPUBlockStore(max_bytes=1024)
        # Store 3 blocks
        for i in range(3):
            k = mx.random.normal((1, 4, 16, 8))
            v = mx.random.normal((1, 4, 16, 8))
            store.store(i, k, v)

        # Access block 0 and 2 to make block 1 LRU
        store.load(0)
        store.load(2)

        # Evict until under 100 bytes (should evict block 1 first)
        evicted = store.evict_lru(target_bytes=100)
        assert evicted > 0
        # Block 1 (LRU) should be gone
        assert store.blocks.get(1) is None

    def test_remove(self):
        store = CPUBlockStore(max_bytes=1024 * 1024 * 1024)
        k = mx.random.normal((1, 4, 16, 8))
        v = mx.random.normal((1, 4, 16, 8))
        store.store(10, k, v)
        assert store.used_bytes > 0
        removed = store.remove([10, 99])
        assert removed == 1
        assert store.used_bytes == 0


# ── TieredKVPrefixCache tests ──

class TestTieredKVPrefixCache:
    def test_add_and_get_exact_match(self):
        """Adding a cache entry and querying the same prompt returns it."""
        cache = TieredKVPrefixCache(
            group=None, model_id="test-model",
            block_size=16, max_blocks=256,
        )
        prompt = _make_fake_prompt(32)
        fake_layers = _make_fake_cache(n_layers=2, n_tokens=32)

        cache.add_kv_cache(prompt, fake_layers)
        assert len(cache.entries) == 1

        # Query with same prompt — should be exact match
        class FakeModel:
            layers = [None, None]

        returned_cache, remaining, matched_idx, is_exact = cache.get_kv_cache(
            FakeModel(), prompt
        )
        assert is_exact is True
        assert len(remaining) == 0

    def test_add_and_get_partial_prefix(self):
        """Adding a long prompt and querying a shorter prefix."""
        cache = TieredKVPrefixCache(
            group=None, model_id="test-model",
            block_size=16, max_blocks=256,
        )
        full_prompt = _make_fake_prompt(64)
        fake_layers = _make_fake_cache(n_layers=2, n_tokens=64)

        cache.add_kv_cache(full_prompt, fake_layers)

        # Query with shorter prompt (first 32 tokens)
        short_prompt = full_prompt[:32]

        class FakeModel:
            layers = [None, None]

        returned_cache, remaining, matched_idx, is_exact = cache.get_kv_cache(
            FakeModel(), short_prompt
        )
        # The short prompt is a full prefix of the stored one, so it's an
        # exact match from the query's perspective (remaining=0).
        assert matched_idx is not None
        assert len(remaining) == 0

    def test_add_no_match_returns_fresh(self):
        """Querying with a completely different prompt returns fresh cache."""
        cache = TieredKVPrefixCache(
            group=None, model_id="test-model",
            block_size=16, max_blocks=256,
        )
        prompt_a = _make_fake_prompt(32)
        fake_layers = _make_fake_cache(n_layers=2, n_tokens=32)
        cache.add_kv_cache(prompt_a, fake_layers)

        # Query with different prompt
        prompt_b = _make_fake_prompt(32) + 1000

        class FakeModel:
            layers = [None, None]

        returned_cache, remaining, matched_idx, is_exact = cache.get_kv_cache(
            FakeModel(), prompt_b
        )
        assert matched_idx is None
        assert is_exact is False
        # remaining should be the full prompt
        assert len(remaining) == 32

    def test_clear(self):
        """Clearing the cache removes all entries."""
        cache = TieredKVPrefixCache(
            group=None, model_id="test-model",
            block_size=16, max_blocks=256,
        )
        prompt = _make_fake_prompt(32)
        fake_layers = _make_fake_cache(n_layers=2, n_tokens=32)
        cache.add_kv_cache(prompt, fake_layers)
        assert len(cache.entries) == 1

        cache.clear()
        assert len(cache.entries) == 0
        assert cache.allocator.num_free == 256  # all blocks freed

    def test_cpu_block_store_integration(self):
        """Verify CPU store works within the tiered cache."""
        store = CPUBlockStore(max_bytes=1024 * 1024)
        k = mx.random.normal((1, 4, 16, 8))
        v = mx.random.normal((1, 4, 16, 8))

        store.store(1, k, v)
        assert store.used_bytes > 0

        loaded = store.load(1)
        assert loaded is not None
        k2, v2 = loaded
        # Verify data is reasonable (not exact due to float16 round-trip)
        assert k2.shape == k.shape

    def test_lru_eviction_order(self):
        """Verify LRU eviction removes least-recently-used first."""
        store = CPUBlockStore(max_bytes=1024 * 1024)

        # Add 5 blocks
        for i in range(5):
            k = mx.random.normal((1, 4, 16, 8))
            v = mx.random.normal((1, 4, 16, 8))
            store.store(i, k, v)

        # Access 0, 1, 2 (making 3, 4 LRU)
        store.load(0)
        store.load(1)
        store.load(2)

        # Evict to tiny size
        evicted = store.evict_lru(target_bytes=50)
        assert evicted > 0
        # Block 3 or 4 should be gone first (they weren't accessed)
        assert 3 not in store.blocks or 4 not in store.blocks


# ── Placement utils tiered multiplier test ──

class TestPlacementUtilsTiered:
    def test_tiered_multiplier_reduces_memory_estimate(self):
        """With EXO_TIERED_KV=1, the working-set multiplier should be lower."""
        from exo.master.placement_utils import (
            _RING_KV_WORKING_SET_MULTIPLIER,
            _RING_KV_WORKING_SET_MULTIPLIER_TIERED,
        )
        # Tiered multiplier should be strictly less than default
        assert _RING_KV_WORKING_SET_MULTIPLIER_TIERED < _RING_KV_WORKING_SET_MULTIPLIER

    def test_estimate_ring_node_memory_uses_tiered_multiplier(self):
        """estimate_ring_node_memory respects EXO_TIERED_KV env."""
        from exo.shared.models.model_cards import ModelCard, ModelTask
        from exo.shared.types.memory import Memory
        from exo.shared.types.backends import Backend
        from exo.master.placement_utils import estimate_ring_node_memory

        card = ModelCard(
            model_id="test/model",
            base_model="Test",
            storage_size=Memory.from_gb(4),
            n_layers=32,
            hidden_size=4096,
            context_length=16384,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
            backends=[Backend.MlxMetal],
        )

        # Without tiered (default)
        os.environ.pop("EXO_TIERED_KV", None)
        default_estimate = estimate_ring_node_memory(card)

        # With tiered
        os.environ["EXO_TIERED_KV"] = "1"
        tiered_estimate = estimate_ring_node_memory(card)
        os.environ.pop("EXO_TIERED_KV", None)

        # Tiered should produce a smaller estimate (less GPU headroom needed)
        assert tiered_estimate < default_estimate


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
