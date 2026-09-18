"""Tests for KV-quantization × rotating-cache interaction (t_7435cb43).

Verifies that EXO_KV_CACHE_BITS never reaches mlx_lm's
maybe_quantize_kv_cache when the cache is a rotating KV cache — mlx_lm's
RotatingKVCache.to_quantized() raises NotImplementedError, so passing
kv_bits with the default rotating cache (T9 #1860, MAX_KV_SIZE=16384)
would crash every generate/prefill call.
"""

from exo.worker.engines.mlx.generator.generate import effective_kv_bits


def test_effective_kv_bits_none_when_kv_bits_none() -> None:
    """kv_bits=None stays None regardless of cache contents."""
    assert effective_kv_bits([], None) is None
    assert effective_kv_bits([object()], None) is None  # type: ignore[list-item]


def test_effective_kv_bits_rotating_cache_disables_quantization() -> None:
    """A rotating cache entry forces kv_bits to None (avoids NYI crash)."""
    from mlx_lm.models.cache import RotatingKVCache

    cache = [RotatingKVCache(max_size=16384, keep=4)]
    assert effective_kv_bits(cache, 4) is None
    assert effective_kv_bits(cache, 8) is None


def test_effective_kv_bits_preserves_bits_for_quantizable_cache() -> None:
    """KVCache (has to_quantized) keeps kv_bits — quantization still works."""
    from mlx_lm.models.cache import KVCache

    cache = [KVCache()]
    assert effective_kv_bits(cache, 4) == 4


def test_effective_kv_bits_empty_cache_keeps_bits() -> None:
    """Empty cache (no evidence of rotation) keeps kv_bits."""
    assert effective_kv_bits([], 4) == 4