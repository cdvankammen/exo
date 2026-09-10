"""Tiered KV cache: GPU -> CPU -> disk.

Three-tier memory management for the KV prefix cache, inspired by vLLM:
- Tier 0 (GPU): BlockAllocator-managed paged blocks on GPU (MX arrays)
- Tier 1 (CPU): Evicted blocks as numpy arrays in system RAM
- Tier 2 (Disk): Persistent safetensors slots on disk

When GPU memory pressure exceeds the threshold, LRU blocks spill to CPU.
When CPU memory pressure exceeds its threshold, old entries flush to disk.
On cache miss, reload checks CPU first (fast), then disk (slow).

Integration: builder.py instantiates TieredKVPrefixCache when EXO_TIERED_KV=1.

Memory accounting:
- GPU usage = block_allocator utilization * block_size * per_element_bytes
- CPU usage = numpy array byte count (capped by EXO_TIERED_CPU_MAX_GB)
- Disk usage = same as existing KVPrefixCache disk path

Reference: harness-hosting-components-2026-09-01.md Section 3.5
Integration: builder.py wraps TieredKVPrefixCache when EXO_TIERED_KV=1
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time as _time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path as _Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import numpy as np

from exo.shared.constants import EXO_CACHE_HOME
from exo.shared.types.memory import Memory
from exo.utils.settings import get_settings_manager
from exo.utils.virtual_memory import virtual_memory_statistics
from exo.worker.engines.mlx.cache import (
    CacheSnapshot,
    KVPrefixCache,
    _bounded_snapshots,
    _partition_cache_state,
    _load_slot_cache,
    cache_length,
    get_prefix_length,
    has_non_kv_caches,
    is_non_trimmable_cache_entry,
    make_kv_cache,
)
from exo.worker.engines.mlx.paged_cache import (
    BlockAllocator,
    PagedKVCache,
    PagedKVPrefixCache,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_MAX_BLOCKS,
)
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from exo.worker.engines.mlx.types import Model
    from exo.worker.engines.mlx.vision import MediaRegion

# ── Environment knobs ──

# Tiered KV is opt-in via EXO_TIERED_KV=1.
_TIERED_KV_ENABLED: bool = os.environ.get("EXO_TIERED_KV", "0") == "1"

# GPU (Metal) eviction threshold: when GPU utilization exceeds this, spill
# LRU blocks to CPU. Must be lower than the general _MEMORY_THRESHOLD.
_GPU_EVICT_THRESHOLD: float = float(
    os.environ.get("EXO_TIERED_GPU_THRESHOLD", "0.75")
)

# CPU tier max size in GB. When CPU usage exceeds this, flush oldest entries
# to disk. Defaults to half of system RAM to leave room for model weights.
def _default_cpu_max_gb() -> float:
    total_gb = virtual_memory_statistics().total_bytes / (1024**3)
    return max(8.0, total_gb * 0.5)

_CPU_MAX_BYTES: int = int(
    float(os.environ.get("EXO_TIERED_CPU_MAX_GB", _default_cpu_max_gb()))
    * 1024
    * 1024
    * 1024
)

# Minimum prefix length to bother searching CPU/disk for.
_MIN_PREFIX_SEARCH: int = int(os.environ.get("EXO_TIERED_MIN_PREFIX", "512"))

# CPU spill flush interval: when a dirty CPU tier entry hasn't been written
# to disk for this many seconds, mark it for disk flush.
_CPU_DISK_FLUSH_INTERVAL: float = 30.0

# ── CPU Block Store ──

# Byte count per element: MX KVCache uses float16 (2 bytes) for K and V.
_BYTES_PER_ELEMENT: int = 2


@dataclass
class CPUBlock:
    """A single KV cache block stored as numpy arrays on CPU."""

    keys: np.ndarray  # shape: (B, n_kv_heads, block_size, k_head_dim)
    values: np.ndarray  # shape: (B, n_kv_heads, block_size, v_head_dim)
    block_id: int  # Original block ID from BlockAllocator (for re-creation)
    last_access: int  # Monotonic access counter for LRU

    @property
    def nbytes(self) -> int:
        return self.keys.nbytes + self.values.nbytes

    @staticmethod
    def from_paged_arrays(
        k: mx.array, v: mx.array, block_id: int, access: int
    ) -> CPUBlock:
        """Create from MLX arrays via numpy round-trip (breaks shared_ptr)."""
        mx.eval(k, v)
        k_np = np.array(k, dtype=np.float16) if k.dtype != mx.bfloat16 else np.array(
            np.array(k.astype(mx.float32)), dtype=np.float16
        )
        v_np = np.array(v, dtype=np.float16) if v.dtype != mx.bfloat16 else np.array(
            np.array(v.astype(mx.float32)), dtype=np.float16
        )
        return CPUBlock(
            keys=k_np, values=v_np, block_id=block_id, last_access=access
        )

    def to_mx(self) -> tuple[mx.array, mx.array]:
        """Restore to MLX arrays."""
        k = mx.array(self.keys)
        v = mx.array(self.values)
        return k, v


@dataclass
class CPUBlockStore:
    """CPU-resident block store with LRU eviction.

    Holds KV cache blocks that have been evicted from GPU memory.  Blocks are
    stored as numpy arrays to avoid consuming Metal/CUDA allocator quota.

    Thread safety: not required — MLX inference is single-threaded per device.
    """

    max_bytes: int = _CPU_MAX_BYTES
    # block_id -> CPUBlock (across all entries, shared pool)
    blocks: dict[int, CPUBlock] = field(default_factory=dict)
    _access_counter: int = 0

    @property
    def used_bytes(self) -> int:
        return sum(b.nbytes for b in self.blocks.values())

    @property
    def utilization(self) -> float:
        if self.max_bytes <= 0:
            return 0.0
        return self.used_bytes / self.max_bytes

    def store(self, block_id: int, k: mx.array, v: mx.array) -> None:
        """Store a block on CPU. Overwrites if block_id already present."""
        self._access_counter += 1
        self.blocks[block_id] = CPUBlock.from_paged_arrays(
            k, v, block_id, self._access_counter
        )

    def load(self, block_id: int) -> tuple[mx.array, mx.array] | None:
        """Load a block from CPU, or None if not present."""
        block = self.blocks.get(block_id)
        if block is None:
            return None
        self._access_counter += 1
        block.last_access = self._access_counter
        return block.to_mx()

    def evict_lru(self, target_bytes: int | None = None) -> int:
        """Evict LRU blocks until under target or max_bytes.

        Returns number of blocks evicted.
        """
        target = target_bytes or self.max_bytes
        evicted = 0
        while self.used_bytes > target and self.blocks:
            # Find least recently used block
            lru_id = min(self.blocks, key=lambda bid: self.blocks[bid].last_access)
            removed = self.blocks.pop(lru_id)
            evicted += 1
        return evicted

    def remove(self, block_ids: list[int]) -> int:
        """Remove specific blocks. Returns count actually removed."""
        removed = 0
        for bid in block_ids:
            if bid in self.blocks:
                del self.blocks[bid]
                removed += 1
        return removed


# ── Tiered KV Cache Entry ──

@dataclass
class TieredCacheEntry:
    """A single conversation's KV cache state across tiers."""

    prompt_tokens: mx.array
    # GPU-resident layers (PagedKVCache per layer)
    gpu_layers: list[PagedKVCache]
    # Block IDs that were spilled to CPU (per-layer lists)
    cpu_spilled_blocks: list[list[int]]
    # Metadata
    last_access: int
    token_count: int
    prefill_tps: float = 0.0
    media_regions: list[Any] = field(default_factory=list)
    # CPU-tier blocks (shared with CPUBlockStore via block IDs)
    cpu_block_ids: list[int] = field(default_factory=list)


# ── Tiered KV Prefix Cache ──

class TieredKVPrefixCache:
    """Three-tier KV cache: GPU -> CPU -> disk.

    Drop-in replacement for KVPrefixCache. When GPU memory pressure is high,
    spills LRU blocks to CPU RAM (numpy arrays). When CPU pressure is high,
    flushes old entries to disk. On cache miss, reloads from CPU first (fast),
    then disk (slow).

    Env gate: EXO_TIERED_KV=1 enables this.
    """

    def __init__(
        self,
        group: Any | None = None,
        model_id: str = "",
        block_size: int = DEFAULT_BLOCK_SIZE,
        max_blocks: int = DEFAULT_MAX_BLOCKS,
    ):
        self._group = group
        self._model_id = model_id

        # GPU tier: shared block allocator
        self.allocator = BlockAllocator(
            block_size=block_size, max_blocks=max_blocks
        )

        # CPU tier: shared block store
        self.cpu_store = CPUBlockStore(max_bytes=_CPU_MAX_BYTES)

        # Cache entries (one per conversation/prefix)
        self.entries: list[TieredCacheEntry] = []

        # LRU tracking
        self._access_counter: int = 0

        # Disk tier: directory per model (same layout as KVPrefixCache)
        self._disk_enabled = (
            get_settings_manager().get_value("EXO_KV_DISK_PERSISTENCE", "0")
        ) == "1"
        self._disk_dir = (
            self._init_disk_dir() if model_id and self._disk_enabled else None
        )
        self._hot_slot_disk_id: int | None = None
        self._flush_requested_at: float = 0.0

        if self._disk_dir is not None:
            self._janitor_sweep_stale_slots()

        logger.info(
            f"TieredKVCache: block_size={block_size}, max_blocks={max_blocks}, "
            f"gpu_threshold={_GPU_EVICT_THRESHOLD:.0%}, "
            f"cpu_max={_CPU_MAX_BYTES / 1024**3:.1f}GB, "
            f"disk={'on' if self._disk_enabled else 'off'}"
        )

    # ── Public API (matches KVPrefixCache) ──

    def clear(self) -> None:
        """Clear all cache entries, releasing GPU and CPU resources."""
        for entry in self.entries:
            for layer in entry.gpu_layers:
                layer.release_all()
        self.cpu_store.remove(
            [bid for entry in self.entries for bid in entry.cpu_block_ids]
        )
        self.entries.clear()

    def add_kv_cache(
        self,
        prompt_tokens: mx.array,
        cache: Any,
        ssm_snapshots: list[CacheSnapshot] | None = None,
        media_regions: list["MediaRegion"] | None = None,
        prefill_tps: float = 0.0,
    ) -> None:
        """Store a completed cache entry. GPU-first, spill LRU if pressured."""
        # GPU eviction check
        self._spill_lru_to_cpu()

        # Create paged layers from the incoming cache
        gpu_layers = self._to_paged_layers(cache)

        self._access_counter += 1
        entry = TieredCacheEntry(
            prompt_tokens=prompt_tokens,
            gpu_layers=gpu_layers,
            cpu_spilled_blocks=[[] for _ in gpu_layers],
            last_access=self._access_counter,
            token_count=len(prompt_tokens),
            prefill_tps=prefill_tps,
            media_regions=media_regions or [],
        )
        self.entries.append(entry)

        logger.info(
            f"TieredKV added: {len(prompt_tokens)} tokens, "
            f"gpu_util={self.allocator.utilization:.1%}, "
            f"cpu_util={self.cpu_store.utilization:.1%}"
        )

    def update_kv_cache(
        self,
        index: int,
        prompt_tokens: mx.array,
        cache: Any,
        snapshots: list[CacheSnapshot] | None,
        restore_pos: int,
        media_regions: list["MediaRegion"] | None = None,
        prefill_tps: float = 0.0,
    ) -> None:
        """Update an existing entry in-place."""
        old = self.entries[index]
        # Release old GPU blocks
        for layer in old.gpu_layers:
            layer.release_all()
        self.cpu_store.remove(old.cpu_block_ids)

        gpu_layers = self._to_paged_layers(cache)

        self._access_counter += 1
        self.entries[index] = TieredCacheEntry(
            prompt_tokens=prompt_tokens,
            gpu_layers=gpu_layers,
            cpu_spilled_blocks=[[] for _ in gpu_layers],
            last_access=self._access_counter,
            token_count=len(prompt_tokens),
            prefill_tps=prefill_tps,
            media_regions=media_regions or [],
        )

    def get_kv_cache(
        self,
        model: "Model",
        prompt_tokens: mx.array,
        media_regions: list["MediaRegion"] | None = None,
    ) -> tuple[Any, mx.array, int | None, bool]:
        """Find best prefix match across GPU, CPU, and disk tiers.

        Returns (cache, remaining_tokens, matched_index, is_exact).
        """
        max_length = len(prompt_tokens)

        # ── Tier 0: GPU prefix match ──
        best_entry: TieredCacheEntry | None = None
        best_length = 0
        is_exact = False

        for i, entry in enumerate(self.entries):
            # GPU-resident tokens cover the stored prompt length
            gpu_token_count = self._entry_gpu_token_count(entry)
            length = get_prefix_length(prompt_tokens, entry.prompt_tokens[:gpu_token_count])
            if length >= max_length - 1:
                best_entry, best_length, is_exact = entry, length, True
                break
            if length > best_length:
                best_entry, best_length = entry, length

        # ── Tier 1: CPU spill match (only if GPU hit was partial) ──
        if best_entry is not None and best_entry.cpu_block_ids:
            gpu_count = self._entry_gpu_token_count(best_entry)
            if best_length < gpu_count:
                # Some blocks are on CPU — promote them back to GPU
                self._promote_cpu_blocks(best_entry)
                gpu_count = self._entry_gpu_token_count(best_entry)
                # Re-check prefix after promotion
                length = get_prefix_length(prompt_tokens, best_entry.prompt_tokens[:gpu_count])
                if length > best_length:
                    best_length = length
                    is_exact = length >= max_length - 1

        # ── Tier 2: Disk fallback ──
        if self._disk_dir:
            if best_entry is not None and best_length < max_length - 1:
                disk_result = self._try_load_from_disk(model, prompt_tokens, min_prefix=best_length)
                if disk_result is not None:
                    return disk_result
            elif best_entry is None:
                disk_result = self._try_load_from_disk(model, prompt_tokens)
                if disk_result is not None:
                    return disk_result

        if best_entry is None:
            return make_kv_cache(model), prompt_tokens, None, False

        # Found match — trim and return
        self._access_counter += 1
        best_entry.last_access = self._access_counter

        gpu_count = self._entry_gpu_token_count(best_entry)
        tokens_to_trim = gpu_count - best_length
        if tokens_to_trim > 0:
            if has_non_kv_caches(best_entry.gpu_layers):
                # Non-trimmable (SSM/Rotating) — need to recompute
                logger.info(
                    f"TieredKV: partial prefix on non-trimmable cache — "
                    f"recomputing"
                )
                return make_kv_cache(model), prompt_tokens, None, False
            # Trim GPU layers
            for layer in best_entry.gpu_layers:
                layer.trim(tokens_to_trim)
            # Also trim CPU spilled blocks beyond the match point
            blocks_in_match = best_length // self.allocator.block_size
            for layer_idx, spilled in enumerate(best_entry.cpu_spilled_blocks):
                excess = spilled[blocks_in_match:]
                if excess:
                    self.cpu_store.remove(excess)
                    best_entry.cpu_spilled_blocks[layer_idx] = spilled[:blocks_in_match]
            # Trim prompt tokens
            best_entry.prompt_tokens = best_entry.prompt_tokens[:best_length]
            best_entry.token_count = best_length

        prompt_cache = copy.deepcopy(best_entry.gpu_layers)
        remaining = prompt_tokens[best_length:]

        return prompt_cache, remaining, best_entry.last_access, is_exact

    def flush_to_disk(self, force: bool = False) -> None:
        """Flush dirty entries to disk if idle long enough."""
        if not self._disk_dir:
            return
        if not force and (_time.time() - self._flush_requested_at) < _CPU_DISK_FLUSH_INTERVAL:
            return
        # Flush the most-recently-used entry to disk
        if self.entries:
            self._flush_entry_to_disk(self.entries[-1])
        self._evict_stale_disk_slots()

    def get_memory_used_percentage(self) -> float:
        """GPU memory pressure (may include distributed all_gather)."""
        # CPU store doesn't directly affect GPU pressure, but
        # total system pressure includes both.
        local_gpu = self.allocator.utilization
        if self._group is None:
            return local_gpu
        all_pressure = mx.distributed.all_gather(
            mx.array([local_gpu], dtype=mx.float32),
            group=self._group,
        )
        return float(mx.max(all_pressure).item())

    def should_update_entry(
        self,
        matched_index: int | None,
        prefix_hit_length: int,
        min_prefix_hit_length: int = 1000,
    ) -> bool:
        """True when the new prompt EXTENDS the matched cached entry."""
        if matched_index is None:
            return False
        # Find entry by last_access id (matched_index is the access counter)
        for entry in self.entries:
            if entry.last_access == matched_index:
                if prefix_hit_length < min_prefix_hit_length:
                    return False
                cached_len = self._entry_gpu_token_count(entry)
                return prefix_hit_length >= cached_len - 8
        return False

    # ── Internal: GPU -> CPU spill ──

    def _spill_lru_to_cpu(self) -> None:
        """Evict GPU blocks from LRU entries to CPU until below threshold."""
        while self.allocator.utilization > _GPU_EVICT_THRESHOLD and self.entries:
            # Find LRU entry
            lru_idx = min(
                range(len(self.entries)),
                key=lambda i: self.entries[i].last_access,
            )
            entry = self.entries[lru_idx]

            # Try to spill one block per layer
            spilled_any = False
            for layer_idx, layer in enumerate(entry.gpu_layers):
                if layer.size() == 0:
                    continue
                # Spill the last block (most recently added, least likely
                # to be needed for prefix matching of future requests)
                last_block_idx = len(layer._block_table) - 1
                if last_block_idx < 0:
                    continue
                bid = layer._block_table[last_block_idx]
                if bid in layer._keys:
                    k, v = layer._keys[bid], layer._values[bid]
                    # Store to CPU
                    self.cpu_store.store(bid, k, v)
                    # Release from GPU
                    layer._block_table.pop()
                    layer._keys.pop(bid, None)
                    layer._values.pop(bid, None)
                    layer.offset = max(0, layer.offset - layer.block_size)
                    entry.cpu_spilled_blocks[layer_idx].append(bid)
                    spilled_any = True

            if not spilled_any:
                # All entries have empty GPU layers — evict the LRU entirely
                self._evict_entry(lru_idx)

    def _evict_entry(self, idx: int) -> None:
        """Remove an entry entirely (GPU + CPU)."""
        entry = self.entries.pop(idx)
        for layer in entry.gpu_layers:
            layer.release_all()
        self.cpu_store.remove(entry.cpu_block_ids)
        logger.info(
            f"TieredKV evicted entry ({entry.token_count} tokens)"
        )

    # ── Internal: CPU -> GPU promotion ──

    def _promote_cpu_blocks(self, entry: TieredCacheEntry) -> None:
        """Move CPU blocks back to GPU for the given entry."""
        for layer_idx, layer in enumerate(entry.gpu_layers):
            cpu_bids = list(entry.cpu_spilled_blocks[layer_idx])
            for bid in cpu_bids:
                data = self.cpu_store.load(bid)
                if data is None:
                    continue
                k_mx, v_mx = data
                # Allocate a fresh GPU block and write the data
                new_blocks = self.allocator.allocate(1)
                new_bid = new_blocks[0]
                # Initialize arrays and write
                if layer._k_head_dim is not None:
                    layer._keys[new_bid] = k_mx
                    layer._values[new_bid] = v_mx
                    layer._block_table.append(new_bid)
                    layer.offset += layer.block_size
                else:
                    # First promotion — need to initialize shapes
                    layer._keys[new_bid] = k_mx
                    layer._values[new_bid] = v_mx
                    layer._block_table.append(new_bid)
                    layer.offset += layer.block_size
            # Clear promoted blocks from CPU store
            self.cpu_store.remove(cpu_bids)
            entry.cpu_spilled_blocks[layer_idx].clear()

        logger.info(
            f"TieredKV promoted CPU blocks to GPU for "
            f"{entry.token_count}-token entry"
        )

    # ── Internal: helpers ──

    def _entry_gpu_token_count(self, entry: TieredCacheEntry) -> int:
        """Number of tokens currently resident on GPU for this entry."""
        if not entry.gpu_layers:
            return 0
        return min(layer.size() for layer in entry.gpu_layers) if entry.gpu_layers else 0

    def _to_paged_layers(self, cache: Any) -> list[PagedKVCache]:
        """Convert any cache type to PagedKVCache layers."""
        if cache and isinstance(cache[0], PagedKVCache):
            return list(cache)
        paged_layers: list[PagedKVCache] = []
        for layer_cache in cache:
            paged = PagedKVCache(self.allocator)
            if hasattr(layer_cache, "keys") and layer_cache.keys is not None:
                keys = layer_cache.keys[..., : layer_cache.offset, :]
                values = layer_cache.values[..., : layer_cache.offset, :]
                if keys.size > 0:
                    paged.update_and_fetch(keys, values)
            elif hasattr(layer_cache, "state"):
                state = layer_cache.state
                if state and len(state) == 2 and state[0] is not None:
                    keys, values = state
                    if keys.size > 0:
                        paged.update_and_fetch(keys, values)
            paged_layers.append(paged)
        return paged_layers

    # ── Internal: Disk persistence (adapted from KVPrefixCache) ──

    def _init_disk_dir(self) -> _Path:
        h = hashlib.sha256(self._model_id.encode()).hexdigest()[:16]
        base = _Path(
            get_settings_manager().get_value(
                "EXO_KV_DISK_PATH", str(EXO_CACHE_HOME / "kv-cache")
            )
        )
        d = base / h
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"TieredKV disk dir: {d}")
        return d

    def _list_disk_slots(self) -> list[int]:
        if self._disk_dir is None:
            return []
        slots: list[int] = []
        for f in self._disk_dir.glob("slot_*_tokens.safetensors"):
            try:
                slots.append(int(f.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return sorted(slots)

    def _next_disk_slot_id(self) -> int:
        existing = self._list_disk_slots()
        return max(existing) + 1 if existing else 0

    def _flush_entry_to_disk(self, entry: TieredCacheEntry) -> None:
        """Flush a single entry to disk (GPU + CPU blocks consolidated)."""
        if self._disk_dir is None:
            return

        # First promote all CPU blocks back to GPU for this entry
        self._promote_cpu_blocks(entry)

        if not entry.gpu_layers:
            return

        try:
            self._disk_dir.mkdir(parents=True, exist_ok=True)
            slot_id = (
                self._hot_slot_disk_id
                if self._hot_slot_disk_id is not None
                else self._next_disk_slot_id()
            )
            base = self._disk_dir / f"slot_{slot_id}"
            cache = list(entry.gpu_layers)
            cache_data, placeholders = _partition_cache_state(cache)
            cache_info: list[Any] = [c.meta_state for c in cache]
            cache_classes = [type(c).__name__ for c in cache]
            cache_metadata: dict[str, str] = {
                k: str(v)
                for k, v in zip(
                    [f"info_{i}" for i in range(len(cache_info))],
                    [json.dumps(info) for info in cache_info],
                )
            }
            cache_metadata.update(
                {f"class_{i}": c for i, c in enumerate(cache_classes)}
            )

            tmp_cache = str(base) + "_tmp_cache.safetensors"
            mx.save_safetensors(tmp_cache, cache_data, cache_metadata)
            os.rename(tmp_cache, str(base) + "_cache.safetensors")

            meta = {
                "model_id": self._model_id,
                "token_count": entry.token_count,
                "timestamp": _time.time(),
                "format": 2,
                "placeholders": placeholders,
            }
            tmp_meta = str(base) + "_meta.json.tmp"
            with open(tmp_meta, "w") as f:
                json.dump(meta, f)
            os.rename(tmp_meta, str(base) + "_meta.json")

            tmp_tokens = str(base) + "_tmp_tokens.safetensors"
            mx.save_safetensors(tmp_tokens, {"tokens": entry.prompt_tokens})
            os.rename(tmp_tokens, str(base) + "_tokens.safetensors")

            self._hot_slot_disk_id = slot_id
            self._flush_requested_at = 0.0
            logger.info(
                f"TieredKV flushed to disk: slot_{slot_id} "
                f"({entry.token_count} tokens)"
            )
        except Exception as e:
            logger.warning(f"TieredKV disk flush failed: {e}")

    def _try_load_from_disk(
        self, model: "Model", prompt_tokens: mx.array, min_prefix: int = 0
    ) -> tuple[Any, mx.array, int, bool] | None:
        """Swap in a matching disk slot."""
        if self._disk_dir is None:
            return None

        best_id: int | None = None
        best_length = 0
        for slot_id in self._list_disk_slots():
            if slot_id == self._hot_slot_disk_id:
                continue
            token_file = self._disk_dir / f"slot_{slot_id}_tokens.safetensors"
            try:
                cached_tokens = mx.load(str(token_file))["tokens"]
                prefix_len = get_prefix_length(prompt_tokens, cached_tokens)
            except Exception:
                continue
            if prefix_len > best_length:
                best_length = prefix_len
                best_id = slot_id

        if best_id is None or best_length <= min_prefix or best_length < _MIN_PREFIX_SEARCH:
            return None

        try:
            base = self._disk_dir / f"slot_{best_id}"
            meta_path = str(base) + "_meta.json"
            placeholders: dict[str, dict[str, Any]] = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    slot_meta = json.load(f)
                placeholders = slot_meta.get("placeholders") or {}
            cache: Any = _load_slot_cache(
                str(base) + "_cache.safetensors", placeholders
            )
            tokens = mx.load(str(base) + "_tokens.safetensors")["tokens"]
            cached_length = cache_length(cache)
            if cached_length <= 0 or cached_length > int(tokens.shape[0]):
                logger.warning(
                    f"TieredKV disk slot_{best_id} inconsistent — skipping"
                )
                return None
            prefix_len = min(best_length, cached_length)
            if prefix_len <= min_prefix:
                return None
            tokens_to_trim = cached_length - prefix_len
            if tokens_to_trim > 0 and has_non_kv_caches(cache):
                logger.info(
                    f"TieredKV disk slot_{best_id}: partial prefix on "
                    f"non-trimmable cache — recomputing"
                )
                return None
            prompt_cache = copy.deepcopy(cache)
        except Exception as e:
            logger.warning(f"TieredKV disk load failed for slot_{best_id}: {e}")
            return None

        # Load succeeded — evict GPU entries to make room
        self._spill_lru_to_cpu()
        self.clear()

        self._access_counter += 1
        gpu_layers = self._to_paged_layers(prompt_cache)
        entry = TieredCacheEntry(
            prompt_tokens=tokens,
            gpu_layers=gpu_layers,
            cpu_spilled_blocks=[[] for _ in gpu_layers],
            last_access=self._access_counter,
            token_count=len(tokens),
        )
        self.entries.append(entry)
        self._hot_slot_disk_id = best_id

        if tokens_to_trim > 0:
            for layer in entry.gpu_layers:
                layer.trim(tokens_to_trim)
            entry.prompt_tokens = entry.prompt_tokens[:prefix_len]
            entry.token_count = prefix_len

        logger.info(
            f"TieredKV loaded from disk: slot_{best_id} ({len(tokens)} tokens)"
        )
        remaining = prompt_tokens[prefix_len:]
        return prompt_cache, remaining, 0, False

    def _janitor_sweep_stale_slots(self) -> None:
        """TTL sweep of other models' slots."""
        if self._disk_dir is None:
            return
        max_age_hours = float(
            get_settings_manager().get_value("EXO_KV_DISK_TTL_HOURS", "24")
        )
        cutoff = _time.time() - (max_age_hours * 3600)
        for meta_file in self._disk_dir.parent.glob("*/slot_*_meta.json"):
            try:
                if meta_file.parent == self._disk_dir:
                    continue
                with open(meta_file) as f:
                    meta = json.load(f)
                if meta.get("timestamp", 0) >= cutoff:
                    continue
                base = str(meta_file)[: -len("_meta.json")]
                for ext in [
                    "_cache.safetensors",
                    "_tokens.safetensors",
                    "_meta.json",
                ]:
                    with open(os.devnull, "w"):
                        try:
                            os.remove(base + ext)
                        except FileNotFoundError:
                            pass
                logger.info(
                    f"TieredKV janitor: evicted stale "
                    f"{meta_file.parent.name}/{meta_file.stem}"
                )
            except Exception:
                continue

    def _evict_stale_disk_slots(self) -> None:
        """Delete stale own-dir slots (TTL), then enforce global size cap."""
        if self._disk_dir is None:
            return
        max_age_hours = float(
            get_settings_manager().get_value("EXO_KV_DISK_TTL_HOURS", "24")
        )
        max_size_gb = float(
            get_settings_manager().get_value("EXO_KV_DISK_MAX_SIZE_GB", "500")
        )
        cutoff = _time.time() - (max_age_hours * 3600)

        for meta_file in self._disk_dir.glob("slot_*_meta.json"):
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                if meta.get("timestamp", 0) < cutoff:
                    slot_id = int(meta_file.stem.split("_")[1])
                    if slot_id == self._hot_slot_disk_id:
                        continue
                    base = self._disk_dir / f"slot_{slot_id}"
                    for ext in [
                        "_cache.safetensors",
                        "_tokens.safetensors",
                        "_meta.json",
                    ]:
                        try:
                            os.remove(str(base) + ext)
                        except FileNotFoundError:
                            pass
                    logger.info(f"TieredKV evicted stale disk slot_{slot_id}")
            except Exception:
                continue

        self._janitor_sweep_stale_slots()

        # Global size cap enforcement
        max_bytes = int(max_size_gb * 1024**3)
        while True:
            slots: list[tuple[float, _Path, int]] = []
            total_size = 0
            for meta_file in self._disk_dir.parent.glob("*/slot_*_meta.json"):
                try:
                    with open(meta_file) as f:
                        meta = json.load(f)
                    slot_id = int(meta_file.stem.split("_")[1])
                    if (
                        meta_file.parent == self._disk_dir
                        and slot_id == self._hot_slot_disk_id
                    ):
                        continue
                    base = meta_file.parent / f"slot_{slot_id}"
                    slot_size = sum(
                        f.stat().st_size
                        for ext in [
                            "_cache.safetensors",
                            "_tokens.safetensors",
                            "_meta.json",
                        ]
                        if (f := _Path(str(base) + ext)).exists()
                    )
                    total_size += slot_size
                    slots.append(
                        (meta.get("timestamp", 0), base, slot_size)
                    )
                except Exception:
                    continue
            if total_size <= max_bytes or not slots:
                break
            slots.sort()
            _oldest_ts, oldest_base, oldest_size = slots[0]
            for ext in [
                "_cache.safetensors",
                "_tokens.safetensors",
                "_meta.json",
            ]:
                try:
                    os.remove(str(oldest_base) + ext)
                except FileNotFoundError:
                    pass
            logger.info(
                f"TieredKV evicted disk slot ({oldest_size / 1024**3:.1f} GB) "
                f"--- kv-cache over {max_size_gb} GB total"
            )


def make_tiered_kv_cache(
    model: "Model",
    group: Any | None = None,
    model_id: str = "",
) -> TieredKVPrefixCache:
    """Factory: create a TieredKVPrefixCache for the given model."""
    return TieredKVPrefixCache(
        group=group,
        model_id=model_id,
    )
