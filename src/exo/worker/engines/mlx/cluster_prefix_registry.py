"""Cluster-Wide Prefix Cache Registry.

Master-side coordination that tracks which workers in the exo cluster hold
cached KV states for which token prefixes. Enables cross-worker prefix
routing: when a worker receives a prompt, it first checks the registry
to see if another worker already has the full prefix cached, allowing
zero-recompute for the shared portion.

Architecture:
- PrefixCacheRegistry: master-side, maintains worker→prefix mapping
- Sync protocol: workers periodically push their cached prefix hashes
- Query: workers ask "who has prefix X?" to get the best match

Reference: harness-hosting-components-2026-09-01.md §3.4
Integration: Master coords via existing mx.distributed.Group.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkerPrefixEntry:
    """A cached prefix on a specific worker."""

    worker_rank: int
    prefix_hash: str  # SHA-256 of token tuple
    tokens_len: int
    layer_start: int  # first layer this worker serves
    layer_end: int  # last layer this worker serves (+1)
    prefill_tps: float = 0.0
    last_reported: float = 0.0  # monotonic timestamp


@dataclass
class ClusterPrefixSnapshot:
    """Snapshot of prefix availability across the cluster."""

    entries: list[WorkerPrefixEntry] = field(default_factory=list)

    def best_worker_for_prefix(
        self, prefix_hash: str, worker_rank: int
    ) -> WorkerPrefixEntry | None:
        """Find the best worker holding this prefix hash.

        Prefers workers in the same layer range (for cache locality),
        then workers with highest prefill TPS.
        """
        candidates = [
            e for e in self.entries if e.prefix_hash == prefix_hash
        ]
        if not candidates:
            return None
        # Prefer same worker (local hit), then highest TPS
        local = [e for e in candidates if e.worker_rank == worker_rank]
        if local:
            return max(local, key=lambda e: e.prefill_tps)
        return max(candidates, key=lambda e: e.prefill_tps)

    def matching_prefixes(
        self, token_hash_prefix: str, worker_rank: int
    ) -> list[WorkerPrefixEntry]:
        """Find workers holding prefixes whose hash starts with a prefix."""
        return [
            e
            for e in self.entries
            if e.prefix_hash.startswith(token_hash_prefix)
        ]


class PrefixCacheRegistry:
    """Master-side registry for cluster-wide prefix cache coordination.

    Tracks which workers have cached which prefix hashes. Workers sync
    their prefix sets periodically; the registry uses this for routing.

    Memory: entries are lightweight (just hash + metadata, not KV data).
    KV data stays on the worker's local GPU; only metadata is shared.
    """

    def __init__(self) -> None:
        # token_hash → list of worker entries
        self._entries: dict[str, list[WorkerPrefixEntry]] = {}
        self._last_sync: dict[int, float] = {}  # rank → last sync time

    @staticmethod
    def hash_tokens(tokens: list[int]) -> str:
        """Deterministic hash for a token sequence."""
        return hashlib.sha256(bytes(tokens)).hexdigest()[:16]

    def sync_worker(
        self,
        worker_rank: int,
        prefixes: list[list[int]],
        layer_start: int = 0,
        layer_end: int = -1,
        prefill_tps: float = 0.0,
        timestamp: float = 0.0,
    ) -> None:
        """Update registry with a worker's current cached prefixes."""
        # Remove old entries for this worker
        for h in list(self._entries.keys()):
            self._entries[h] = [
                e for e in self._entries[h] if e.worker_rank != worker_rank
            ]
            if not self._entries[h]:
                del self._entries[h]

        # Add new entries
        for prefix_tokens in prefixes:
            h = self.hash_tokens(prefix_tokens)
            if h not in self._entries:
                self._entries[h] = []
            self._entries[h].append(
                WorkerPrefixEntry(
                    worker_rank=worker_rank,
                    prefix_hash=h,
                    tokens_len=len(prefix_tokens),
                    layer_start=layer_start,
                    layer_end=layer_end,
                    prefill_tps=prefill_tps,
                    last_reported=timestamp,
                )
            )

        self._last_sync[worker_rank] = timestamp

    def lookup(
        self, token_hash: str, worker_rank: int
    ) -> WorkerPrefixEntry | None:
        """Find the best worker for a given prefix hash."""
        snapshot = ClusterPrefixSnapshot(
            entries=[
                e
                for elist in self._entries.values()
                for e in elist
            ]
        )
        return snapshot.best_worker_for_prefix(token_hash, worker_rank)

    def worker_entry_count(self, worker_rank: int) -> int:
        return sum(
            1
            for elist in self._entries.values()
            for e in elist
            if e.worker_rank == worker_rank
        )

    @property
    def total_prefixes(self) -> int:
        return len(self._entries)

    def stats(self) -> dict[str, Any]:
        return {
            "total_prefixes": self.total_prefixes,
            "workers_synced": len(self._last_sync),
            "last_syncs": dict(self._last_sync),
        }
