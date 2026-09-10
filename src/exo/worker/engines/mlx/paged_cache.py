"""PagedAttention-style KV cache for MLX.

Implements vLLM-inspired block-level memory management for the KV cache:
- BlockAllocator: manages a pool of fixed-size blocks (default 16 tokens each)
- PagedKVCache: per-layer cache that maps logical token positions to physical blocks
- PagedKVPrefixCache: drop-in replacement for KVPrefixCache using paged allocation

Reference: model-distribution-strategies-2026-09-01.md, recommendations.md P2.3
Integration: cache.py wraps PagedKVCache via make_kv_cache when enabled.

Memory efficiency gain: eliminates internal fragmentation from step-based
pre-allocation (KVCache.step=256). A 7-token sequence uses 1 block (16 tokens)
instead of 256 tokens of pre-allocated space. At scale across many concurrent
requests this yields ~2x memory efficiency.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mlx.core as mx

from exo.utils.settings import get_settings_manager
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from exo.worker.engines.mlx.types import Model

# Default block size in tokens. 16 matches vLLM's default and aligns well
# with Apple Silicon page sizes for unified memory.
DEFAULT_BLOCK_SIZE: int = int(os.environ.get("EXO_PAGED_BLOCK_SIZE", "16"))

# Maximum number of blocks in the allocator pool. Controls total KV memory cap.
# Each block holds (n_kv_heads, block_size, head_dim) floats for K and V.
DEFAULT_MAX_BLOCKS: int = int(os.environ.get("EXO_PAGED_MAX_BLOCKS", "8192"))


@dataclass
class BlockAllocator:
    """Manages a pool of fixed-size KV cache blocks.

    Blocks are integer indices into a shared physical memory pool. The allocator
    tracks free vs allocated blocks and supports reference counting so that
    prefix sharing between requests (e.g. shared system prompts) is zero-copy.

    Thread safety: not required — MLX inference is single-threaded per device.
    """

    block_size: int = DEFAULT_BLOCK_SIZE
    max_blocks: int = DEFAULT_MAX_BLOCKS

    # Free list: stack of available block indices
    _free_blocks: list[int] = field(default_factory=list, repr=False)
    # Reference counts for allocated blocks
    _ref_counts: dict[int, int] = field(default_factory=dict, repr=False)
    # Total allocated count for diagnostics
    _allocated_count: int = 0

    def __post_init__(self):
        if not self._free_blocks:
            # Initialize all blocks as free (highest index first so low indices
            # are allocated first, improving cache locality).
            self._free_blocks = list(range(self.max_blocks - 1, -1, -1))

    @property
    def num_free(self) -> int:
        return len(self._free_blocks)

    @property
    def num_allocated(self) -> int:
        return self._allocated_count

    @property
    def utilization(self) -> float:
        if self.max_blocks == 0:
            return 0.0
        return self._allocated_count / self.max_blocks

    def allocate(self, num_blocks: int = 1) -> list[int]:
        """Allocate `num_blocks` physical blocks. Returns block indices.

        Raises RuntimeError if insufficient free blocks.
        """
        if num_blocks > len(self._free_blocks):
            raise RuntimeError(
                f"BlockAllocator OOM: requested {num_blocks} blocks, "
                f"only {len(self._free_blocks)} free (total={self.max_blocks})"
            )
        blocks = []
        for _ in range(num_blocks):
            block_id = self._free_blocks.pop()
            self._ref_counts[block_id] = 1
            blocks.append(block_id)
        self._allocated_count += num_blocks
        return blocks

    def free(self, block_ids: list[int]) -> None:
        """Decrement ref counts; return fully-released blocks to the free list."""
        for bid in block_ids:
            if bid not in self._ref_counts:
                continue
            self._ref_counts[bid] -= 1
            if self._ref_counts[bid] <= 0:
                del self._ref_counts[bid]
                self._free_blocks.append(bid)
                self._allocated_count -= 1

    def ref(self, block_ids: list[int]) -> None:
        """Increment reference counts for shared prefix blocks."""
        for bid in block_ids:
            if bid in self._ref_counts:
                self._ref_counts[bid] += 1

    def reset(self) -> None:
        """Release all blocks back to the free pool."""
        self._free_blocks = list(range(self.max_blocks - 1, -1, -1))
        self._ref_counts.clear()
        self._allocated_count = 0


class PagedKVCache:
    """Per-layer paged KV cache implementing the _BaseCache interface.

    Instead of a single contiguous array that grows in large steps (like
    mlx_lm's KVCache with step=256), this stores KV data in fixed-size
    blocks managed by a shared BlockAllocator. Logical token positions
    map to physical block indices through a block table.

    Shape convention per block:
        keys[block_id]:   (B, n_kv_heads, block_size, k_head_dim)
        values[block_id]: (B, n_kv_heads, block_size, v_head_dim)

    This class is designed to be a drop-in replacement for mlx_lm's KVCache
    in exo's generation pipeline. It implements update_and_fetch, trim,
    state/meta_state properties, and make_mask.
    """

    def __init__(
        self,
        allocator: BlockAllocator,
        block_size: int | None = None,
    ):
        self.allocator = allocator
        self.block_size = block_size or allocator.block_size
        self.offset: int = 0  # total logical tokens stored
        self._block_table: list[int] = []  # logical block idx -> physical block id

        # Physical storage: block_id -> mx.array
        # Lazily populated on first update_and_fetch call when we know shapes.
        self._keys: dict[int, mx.array] = {}
        self._values: dict[int, mx.array] = {}

        # Shape metadata (set on first write)
        self._k_head_dim: int | None = None
        self._v_head_dim: int | None = None
        self._n_kv_heads: int | None = None
        self._batch: int | None = None
        self._dtype: mx.Dtype | None = None

    def _ensure_block(self, logical_idx: int) -> int:
        """Ensure a physical block exists for the given logical block index."""
        while logical_idx >= len(self._block_table):
            new_blocks = self.allocator.allocate(1)
            bid = new_blocks[0]
            self._block_table.append(bid)
            # Allocate zero-initialized arrays for this block
            if self._k_head_dim is not None:
                self._keys[bid] = mx.zeros(
                    (self._batch, self._n_kv_heads, self.block_size, self._k_head_dim),
                    dtype=self._dtype,
                )
                self._values[bid] = mx.zeros(
                    (self._batch, self._n_kv_heads, self.block_size, self._v_head_dim),
                    dtype=self._dtype,
                )
        return self._block_table[logical_idx]

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """Append new KV entries and return full cache up to current offset.

        Args:
            keys:   (B, n_kv_heads, S, k_head_dim) new key states
            values: (B, n_kv_heads, S, v_head_dim) new value states

        Returns:
            (all_keys, all_values) each shaped (B, n_kv_heads, offset+S, head_dim)
        """
        B, n_kv_heads, S, k_head_dim = keys.shape
        v_head_dim = values.shape[3]

        # Initialize shape metadata on first call
        if self._k_head_dim is None:
            self._k_head_dim = k_head_dim
            self._v_head_dim = v_head_dim
            self._n_kv_heads = n_kv_heads
            self._batch = B
            self._dtype = keys.dtype

        prev_offset = self.offset
        end_offset = prev_offset + S

        # Calculate which blocks we need
        start_block = prev_offset // self.block_size
        end_block = (end_offset - 1) // self.block_size

        # Ensure all needed blocks are allocated
        for logical_idx in range(start_block, end_block + 1):
            bid = self._ensure_block(logical_idx)
            # Write the portion of keys/values that falls into this block
            block_start_token = logical_idx * self.block_size
            write_start = max(0, prev_offset - block_start_token)
            write_end = min(self.block_size, end_offset - block_start_token)
            src_start = max(0, block_start_token - prev_offset)
            src_end = src_start + (write_end - write_start)

            if write_end > write_start:
                self._keys[bid] = self._keys[bid].at[..., write_start:write_end, :].add(
                    keys[..., src_start:src_end, :] * 0  # placeholder: use direct assignment below
                )
                # MLX doesn't have in-place slice assignment like numpy.
                # We reconstruct the block with the new data inserted.
                old_k = self._keys[bid]
                old_v = self._values[bid]
                parts_k = []
                parts_v = []
                if write_start > 0:
                    parts_k.append(old_k[..., :write_start, :])
                    parts_v.append(old_v[..., :write_start, :])
                parts_k.append(keys[..., src_start:src_end, :])
                parts_v.append(values[..., src_start:src_end, :])
                if write_end < self.block_size:
                    parts_k.append(old_k[..., write_end:, :])
                    parts_v.append(old_v[..., write_end:, :])
                self._keys[bid] = mx.concatenate(parts_k, axis=-2)
                self._values[bid] = mx.concatenate(parts_v, axis=-2)

        self.offset = end_offset

        # Gather all blocks into contiguous output
        return self._gather(self.offset)

    def _gather(self, length: int) -> tuple[mx.array, mx.array]:
        """Gather physical blocks into a contiguous (B, heads, length, dim) tensor."""
        if length == 0 or not self._block_table:
            return mx.zeros((self._batch or 1, self._n_kv_heads or 1, 0, self._k_head_dim or 1)), \
                   mx.zeros((self._batch or 1, self._n_kv_heads or 1, 0, self._v_head_dim or 1))

        num_full_blocks = length // self.block_size
        remainder = length % self.block_size

        k_parts = []
        v_parts = []
        for i in range(num_full_blocks):
            bid = self._block_table[i]
            k_parts.append(self._keys[bid])
            v_parts.append(self._values[bid])
        if remainder > 0:
            bid = self._block_table[num_full_blocks]
            k_parts.append(self._keys[bid][..., :remainder, :])
            v_parts.append(self._values[bid][..., :remainder, :])

        if not k_parts:
            return mx.zeros((self._batch, self._n_kv_heads, 0, self._k_head_dim), dtype=self._dtype), \
                   mx.zeros((self._batch, self._n_kv_heads, 0, self._v_head_dim), dtype=self._dtype)

        return mx.concatenate(k_parts, axis=-2), mx.concatenate(v_parts, axis=-2)

    def size(self) -> int:
        return self.offset

    @property
    def state(self) -> tuple[list[int], int]:
        """Serializable state: block table + offset."""
        return (list(self._block_table), self.offset)

    @state.setter
    def state(self, v: Any) -> None:
        block_table, offset = v
        self._block_table = list(block_table)
        self.offset = offset

    @property
    def meta_state(self) -> str:
        return str(self.offset)

    @meta_state.setter
    def meta_state(self, v: str) -> None:
        self.offset = int(v)

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        """Remove the last n tokens, freeing whole blocks that become empty."""
        n = min(n, self.offset)
        if n == 0:
            return 0
        new_offset = self.offset - n
        new_num_blocks = (new_offset + self.block_size - 1) // self.block_size if new_offset > 0 else 0
        old_num_blocks = len(self._block_table)

        # Free blocks beyond what's needed
        blocks_to_free = self._block_table[new_num_blocks:]
        if blocks_to_free:
            self.allocator.free(blocks_to_free)
            for bid in blocks_to_free:
                self._keys.pop(bid, None)
                self._values.pop(bid, None)

        self._block_table = self._block_table[:new_num_blocks]
        self.offset = new_offset
        return n

    def make_mask(self, N: int, window_size: int | None = None, return_array: bool = False):
        """Generate causal attention mask compatible with the cache offset."""
        from mlx_lm.models.base import create_causal_mask
        if N == 1:
            return None
        if window_size is not None or return_array:
            return create_causal_mask(N, self.offset, window_size=window_size)
        return "causal"

    def empty(self) -> bool:
        return self.offset == 0

    @property
    def nbytes(self) -> int:
        total = 0
        for bid in self._block_table:
            if bid in self._keys:
                total += self._keys[bid].nbytes + self._values[bid].nbytes
        return total

    def release_all(self) -> None:
        """Free all blocks back to the allocator."""
        if self._block_table:
            self.allocator.free(self._block_table)
        self._block_table.clear()
        self._keys.clear()
        self._values.clear()
        self.offset = 0

    def __deepcopy__(self, memo):
        """Custom deepcopy: mx.Dtype and mx.array can't be pickled.

        Creates a new PagedKVCache sharing the same allocator but with
        independent block table and numpy-round-tripped array copies.
        """
        import numpy as np
        cls = self.__class__
        obj = cls.__new__(cls)
        memo[id(self)] = obj
        obj.allocator = self.allocator
        obj.block_size = self.block_size
        obj.offset = self.offset
        obj._block_table = list(self._block_table)
        obj._k_head_dim = self._k_head_dim
        obj._v_head_dim = self._v_head_dim
        obj._n_kv_heads = self._n_kv_heads
        obj._batch = self._batch
        obj._dtype = self._dtype
        # Copy arrays via numpy round-trip to break mlx shared_ptr
        obj._keys = {}
        obj._values = {}
        for bid, arr in self._keys.items():
            obj._keys[bid] = mx.array(np.array(arr))
        for bid, arr in self._values.items():
            obj._values[bid] = mx.array(np.array(arr))
        return obj

    @classmethod
    def from_state(cls, state: Any, meta_state: str) -> "PagedKVCache":
        """Reconstruct from serialized state. Note: requires an allocator to be
        set externally after construction since allocators are shared resources."""
        obj = cls.__new__(cls)
        obj.state = state
        obj.meta_state = meta_state
        obj._keys = {}
        obj._values = {}
        obj._k_head_dim = None
        obj._v_head_dim = None
        obj._n_kv_heads = None
        obj._batch = None
        obj._dtype = None
        return obj


class PagedKVPrefixCache:
    """Drop-in wrapper around KVPrefixCache logic that uses paged allocation.

    Provides the same interface as KVPrefixCache (add_kv_cache, get_kv_cache,
    evict_if_needed) but backed by BlockAllocator + PagedKVCache for ~2x
    memory efficiency over the standard step-growth approach.

    Integration point: builder.py instantiates this instead of KVPrefixCache
    when EXO_PAGED_CACHE=1 is set.
    """

    def __init__(
        self,
        group: Any | None = None,
        model_id: str = "",
        block_size: int = DEFAULT_BLOCK_SIZE,
        max_blocks: int = DEFAULT_MAX_BLOCKS,
    ):
        self.allocator = BlockAllocator(block_size=block_size, max_blocks=max_blocks)
        self.prompts: list[mx.array] = []
        self.caches: list[list[PagedKVCache]] = []
        self._last_used: list[int] = []
        self._access_counter: int = 0
        self._group = group
        self._model_id = model_id

        # Memory thresholds mirror cache.py, routed through SettingsManager.
        _mem_raw = get_settings_manager().get_value("EXO_MEMORY_THRESHOLD", "0.80")
        self._memory_threshold = float(_mem_raw)

    def clear(self) -> None:
        for cache_layers in self.caches:
            for layer_cache in cache_layers:
                layer_cache.release_all()
        self.prompts.clear()
        self.caches.clear()
        self._last_used.clear()

    def add_kv_cache(
        self,
        prompt_tokens: mx.array,
        cache: Any,
        ssm_snapshots: Any = None,
        media_regions: Any = None,
        prefill_tps: float = 0.0,
    ) -> None:
        """Store a completed KV cache entry using paged allocation."""
        self._evict_if_needed()
        self.prompts.append(prompt_tokens)
        # If incoming cache is already PagedKVCache layers, store directly.
        # Otherwise, migrate from standard mlx_lm cache types.
        if cache and isinstance(cache[0], PagedKVCache):
            paged_layers = list(cache)
        else:
            paged_layers = self._migrate_to_paged(cache)
        self.caches.append(paged_layers)
        self._access_counter += 1
        self._last_used.append(self._access_counter)
        logger.info(
            f"PagedKV cache added: {len(prompt_tokens)} tokens, "
            f"allocator utilization={self.allocator.utilization:.1%}"
        )

    def _migrate_to_paged(self, cache: Any) -> list[PagedKVCache]:
        """Convert standard mlx_lm KV cache layers to PagedKVCache.

        Reads the contiguous arrays from each layer's state and writes them
        into newly allocated paged blocks.
        """
        paged_layers: list[PagedKVCache] = []
        for layer_cache in cache:
            paged = PagedKVCache(self.allocator)
            if hasattr(layer_cache, 'keys') and layer_cache.keys is not None:
                keys = layer_cache.keys[..., :layer_cache.offset, :]
                values = layer_cache.values[..., :layer_cache.offset, :]
                if keys.size > 0:
                    paged.update_and_fetch(keys, values)
            elif hasattr(layer_cache, 'state'):
                state = layer_cache.state
                if state and len(state) == 2 and state[0] is not None:
                    keys, values = state
                    if keys.size > 0:
                        paged.update_and_fetch(keys, values)
            paged_layers.append(paged)
        return paged_layers

    def get_kv_cache(
        self,
        model: "Model",
        prompt_tokens: mx.array,
        media_regions: Any = None,
    ) -> tuple[Any, mx.array, int | None, bool]:
        """Find best prefix match and return (cache, remaining_tokens, matched_index, is_exact).

        Uses the same prefix-matching logic as KVPrefixCache.get_kv_cache.
        """
        from exo.worker.engines.mlx.cache import get_prefix_length, make_kv_cache

        max_length = len(prompt_tokens)
        best_index: int | None = None
        best_length = 0
        is_exact = False

        for i, cached_prompt in enumerate(self.prompts):
            length = get_prefix_length(prompt_tokens, cached_prompt)
            if length >= max_length - 1:
                best_index, best_length = i, length
                is_exact = True
                break
            if length > best_length:
                best_index, best_length = i, length

        if best_index is None:
            return make_kv_cache(model), prompt_tokens, None, False

        self._access_counter += 1
        self._last_used[best_index] = self._access_counter

        desired = (max_length - 1) if is_exact else best_length
        target = min(self.caches[best_index][0].offset, desired)
        remaining = prompt_tokens[target:]

        # Return a deep copy of the paged cache layers
        import copy
        prompt_cache = copy.deepcopy(self.caches[best_index])
        return prompt_cache, remaining, best_index, is_exact

    def _evict_if_needed(self) -> None:
        """LRU eviction when allocator utilization exceeds threshold."""
        while (
            self.caches
            and self.allocator.utilization > self._memory_threshold
        ):
            lru_index = self._last_used.index(min(self._last_used))
            evicted_tokens = len(self.prompts[lru_index])
            # Release paged blocks
            for layer_cache in self.caches[lru_index]:
                layer_cache.release_all()
            self.prompts.pop(lru_index)
            self.caches.pop(lru_index)
            self._last_used.pop(lru_index)
            logger.info(
                f"PagedKV evicted LRU entry ({evicted_tokens} tokens), "
                f"utilization={self.allocator.utilization:.1%}"
            )

    def get_memory_used_percentage(self) -> float:
        """Return allocator utilization as a fraction [0, 1]."""
        return self.allocator.utilization


def make_paged_kv_cache(
    model: "Model",
    allocator: BlockAllocator | None = None,
) -> list[PagedKVCache]:
    """Create a paged KV cache for all model layers.

    Drop-in replacement for cache.make_kv_cache when EXO_PAGED_CACHE=1.
    """
    assert hasattr(model, "layers")
    if allocator is None:
        allocator = BlockAllocator()
    return [PagedKVCache(allocator) for _ in model.layers]
