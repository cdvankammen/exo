"""Cluster-Wide Prefix Cache using Radix Tree.

Implements a SGLang-inspired radix tree for efficient prefix matching across
the exo cluster. The radix tree enables O(prefix_length) lookup instead of
O(n * prompt_length) linear scan, and supports hierarchical prefix sharing
between requests with common system prompts or conversation histories.

Architecture:
- RadixNode: tree node storing token chunks and cache references
- ClusterRadixCache: worker-local radix tree with cluster coordination hooks
- PrefixCacheRegistry: master-side registry tracking which workers hold which prefixes

Reference: SGLang unified_radix_cache.py, harness-hosting-components-2026-09-01.md §3.4
Integration: Extends KVPrefixCache (cache.py) with radix-based lookup.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mlx.core as mx

from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from exo.worker.engines.mlx.cache import KVCacheType, CacheSnapshot
    from exo.worker.engines.mlx.vision import MediaRegion


@dataclass
class RadixNode:
    """A node in the radix tree.

    Each node stores a chunk of tokens (the edge label from parent to this node)
    and optionally a cached KV state if this node represents a complete prefix
    that has been prefilled.
    """

    # Token chunk for the edge from parent to this node
    tokens: list[int] = field(default_factory=list)

    # Children keyed by first token of their edge
    children: dict[int, RadixNode] = field(default_factory=dict)

    # Cached KV state at this node (None if not yet prefilled)
    cache: KVCacheType | None = None
    snapshots: list[CacheSnapshot] | None = None
    media_regions: list[MediaRegion] | None = None
    prefill_tps: float = 0.0

    # LRU metadata
    last_access: float = 0.0
    ref_count: int = 0

    # Parent reference for eviction walks
    parent: RadixNode | None = None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def has_cache(self) -> bool:
        return self.cache is not None

    @property
    def depth(self) -> int:
        """Total token depth from root."""
        d = len(self.tokens)
        node = self.parent
        while node is not None:
            d += len(node.tokens)
            node = node.parent
        return d


class ClusterRadixCache:
    """Worker-local radix tree for prefix caching with cluster coordination.

    Provides O(prefix_length) prefix matching vs O(n * prompt_length) for
    the flat list approach in KVPrefixCache. Supports:
    - Hierarchical prefix sharing (system prompt → conversation turns)
    - LRU eviction when memory pressure exceeds threshold
    - Cluster-aware: reports cached prefixes to master for cross-worker routing
    """

    def __init__(
        self,
        group: mx.distributed.Group | None = None,
        model_id: str = "",
        max_nodes: int = 4096,
    ):
        self.root = RadixNode()
        self._group = group
        self._model_id = model_id
        self._max_nodes = max_nodes
        self._node_count = 1  # root
        self._access_counter = 0

        # Stats
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def node_count(self) -> int:
        return self._node_count

    def match_prefix(
        self, prompt_tokens: list[int]
    ) -> tuple[RadixNode | None, int]:
        """Find the deepest node matching a prefix of prompt_tokens.

        Returns:
            (matched_node, matched_length): The deepest cached node and
            the number of tokens matched.
        """
        node = self.root
        pos = 0
        best_node: RadixNode | None = None
        best_length = 0

        while pos < len(prompt_tokens):
            token = prompt_tokens[pos]
            if token not in node.children:
                break

            child = node.children[token]
            edge_tokens = child.tokens
            edge_len = len(edge_tokens)

            # Check how many tokens match along this edge
            match_len = 0
            for i in range(edge_len):
                if pos + i >= len(prompt_tokens):
                    break
                if prompt_tokens[pos + i] != edge_tokens[i]:
                    break
                match_len += 1

            if match_len == 0:
                break

            pos += match_len

            if match_len == edge_len:
                # Full edge matched — descend into child
                if child.has_cache:
                    best_node = child
                    best_length = pos
                node = child
            else:
                # Partial edge match — can't use this node's cache
                break

        if best_node is not None:
            self._hits += 1
            best_node.last_access = time.monotonic()
            best_node.ref_count += 1
        else:
            self._misses += 1

        return best_node, best_length

    def insert(
        self,
        prompt_tokens: list[int],
        cache: KVCacheType,
        snapshots: list[CacheSnapshot] | None = None,
        media_regions: list[MediaRegion] | None = None,
        prefill_tps: float = 0.0,
    ) -> RadixNode:
        """Insert a completed prefix into the radix tree.

        Creates intermediate nodes as needed via edge splitting.
        """
        node = self.root
        pos = 0

        while pos < len(prompt_tokens):
            token = prompt_tokens[pos]

            if token not in node.children:
                # Create new leaf with remaining tokens
                new_node = RadixNode(
                    tokens=prompt_tokens[pos:],
                    parent=node,
                )
                node.children[token] = new_node
                self._node_count += 1
                node = new_node
                break

            child = node.children[token]
            edge_tokens = child.tokens
            edge_len = len(edge_tokens)

            # Find divergence point
            match_len = 0
            for i in range(edge_len):
                if pos + i >= len(prompt_tokens):
                    break
                if prompt_tokens[pos + i] != edge_tokens[i]:
                    break
                match_len += 1

            if match_len == edge_len:
                # Full edge matched — continue down
                pos += match_len
                node = child
            else:
                # Edge split needed
                # Create intermediate node at divergence point
                mid_node = RadixNode(
                    tokens=edge_tokens[:match_len],
                    parent=node,
                )
                # Re-parent existing child
                child.tokens = edge_tokens[match_len:]
                child.parent = mid_node
                mid_node.children[child.tokens[0]] = child
                node.children[token] = mid_node
                self._node_count += 1

                # Create new branch for remaining tokens
                remaining = prompt_tokens[pos + match_len :]
                if remaining:
                    new_node = RadixNode(
                        tokens=remaining,
                        parent=mid_node,
                    )
                    mid_node.children[remaining[0]] = new_node
                    self._node_count += 1
                    node = new_node
                else:
                    node = mid_node
                break

        # Attach cache at terminal node
        node.cache = cache
        node.snapshots = snapshots
        node.media_regions = media_regions
        node.prefill_tps = prefill_tps
        node.last_access = time.monotonic()
        node.ref_count = 1

        # Evict if over capacity
        if self._node_count > self._max_nodes:
            self._evict_lru()

        return node

    def _evict_lru(self) -> None:
        """Evict least-recently-used leaf nodes until under capacity."""
        while self._node_count > self._max_nodes:
            # Find LRU leaf
            lru_node: RadixNode | None = None
            lru_time = float("inf")

            def _find_lru(n: RadixNode) -> None:
                nonlocal lru_node, lru_time
                if n.is_leaf and n is not self.root:
                    if n.last_access < lru_time:
                        lru_time = n.last_access
                        lru_node = n
                for child in n.children.values():
                    _find_lru(child)

            _find_lru(self.root)

            if lru_node is None or lru_node.parent is None:
                break

            # Remove leaf
            parent = lru_node.parent
            first_token = lru_node.tokens[0] if lru_node.tokens else None
            if first_token is not None and first_token in parent.children:
                del parent.children[first_token]
            self._node_count -= 1

            # Merge parent with single remaining child if parent has no cache
            if (
                not parent.has_cache
                and len(parent.children) == 1
                and parent is not self.root
            ):
                only_child = next(iter(parent.children.values()))
                merged_tokens = parent.tokens + only_child.tokens
                only_child.tokens = merged_tokens
                only_child.parent = parent.parent
                if parent.parent is not None:
                    first = merged_tokens[0]
                    parent.parent.children[first] = only_child
                self._node_count -= 1

    def clear(self) -> None:
        """Reset the radix tree."""
        self.root = RadixNode()
        self._node_count = 1
        self._hits = 0
        self._misses = 0

    def get_cached_prefixes(self) -> list[list[int]]:
        """Return all cached prefix token sequences (for cluster registry sync)."""
        result: list[list[int]] = []

        def _collect(node: RadixNode, prefix: list[int]) -> None:
            current = prefix + node.tokens
            if node.has_cache:
                result.append(current)
            for child in node.children.values():
                _collect(child, current)

        for child in self.root.children.values():
            _collect(child, [])
        return result

    def stats(self) -> dict[str, Any]:
        return {
            "node_count": self._node_count,
            "max_nodes": self._max_nodes,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "model_id": self._model_id,
        }
