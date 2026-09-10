import os

from exo.utils.settings import get_settings_manager

# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?

ATTENTION_KV_BITS: int | None = 4
MAX_TOKENS: int = 32168
# T9 (#1860): RotatingKVCache cap. Previously a dead constant (3200) that was
# never wired into the generator, so long conversations grew the KV cache
# until OOM. Now used by make_kv_cache; default 16384 (4x the old cap, still
# bounded) is safer than the PR's 131072 under 8-way parallel batches.
# Power users can raise it via EXO_MAX_KV_SIZE (matches EXO_KV_CACHE_BITS
# knob philosophy).
MAX_KV_SIZE: int | None = int(os.getenv("EXO_MAX_KV_SIZE", "16384"))
KEEP_KV_SIZE: int | None = int(os.getenv("EXO_KEEP_KV_SIZE", "8000"))
QUANTIZE_MODEL_MODE: str | None = "affine"

# Number of bits to quantize the KV cache to (mlx_lm's QuantizedKVCache, e.g.
# 4 or 8). None (the default) keeps the cache in full precision. Opt-in via
# env var: this path is wired into every generation code path here (both
# make_kv_cache's direct QuantizedKVCache construction and mlx_lm's
# maybe_quantize_kv_cache during stream_generate/pipeline prefill), and
# exo's prefix-cache trim/snapshot logic (cache.py) already handles
# QuantizedKVCache correctly via the shared _BaseCache.trim()/.offset
# interface. What's unverified is real-hardware behavior -- turn on only
# after testing on an actual cluster.
# Routed through SettingsManager: persisted UI override > env > in-code default.
# The in-code default stays None (quantization off) — the settings-UI catalog
# default is display-only until an override is actually saved.
_kv_bits_raw = get_settings_manager().get_value("EXO_KV_CACHE_BITS")
KV_CACHE_BITS: int | None = int(_kv_bits_raw) if _kv_bits_raw is not None else None
KV_CACHE_GROUP_SIZE: int = int(
    get_settings_manager().get_value("EXO_KV_CACHE_GROUP_SIZE", "64")
)

DEFAULT_TOP_LOGPROBS: int = 5

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
