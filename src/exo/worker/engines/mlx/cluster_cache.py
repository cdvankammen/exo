"""Cluster-wide prefix cache coordination.

This module implements the *advisory index* half of a cluster-wide prefix
cache for exo: compact token-prefix descriptors (a few KB) are shared across
nodes over the existing event/topic plane; KV tensors never cross the wire in
v1 (mlx arrays pickle but require a copy, and zero-copy mmap sharing is a
research-grade change -- see findings
00-CLUSTER-WIDE-PREFIX-CACHE-t_00968166-ClusterWidePrefixCache.md).

Design (per card t_e5557199, "topic-based coordination consistent with
event-sourced architecture"):
- Each node keeps a local :class:`ClusterPrefixIndex` -- a hash map from
  ``(model_hash, prefix_chunks)`` to peer descriptors.
- On ``add_kv_cache`` / ``update_kv_cache`` a runner *publishes* a
  :class:`PrefixIndexEvent` (compact: model_hash, token prefix hash
  chunks, lengths, node_id, timestamp) via its event channel.
- On ``get_kv_cache`` miss it *consults* the local index; if a peer holds the
  prefix, it logs a ``PREFIX_CLUSTER_HIT`` event (hit length + owning node)
  and falls back to per-instance prefill of only the delta. v1 is advisory:
  no cross-node KV tensor fetch.

Everything is env-gated behind ``EXO_CLUSTER_PREFIX_CACHE=1`` (default off),
so the default behavior is byte-identical to today.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from exo.shared.types.events import PrefixIndexEvent

from exo.worker.runner.bootstrap import logger

# ---------------------------------------------------------------------------
# Compact descriptor model
# ---------------------------------------------------------------------------

_PREFIX_CHUNK_TOKENS = 256
"""Token granularity of prefix chunks. 256 tokens per hash chunk keeps the
descriptor tiny (32 bytes/chunk at sha256) while still giving useful partial
matches for multi-hundred-token prompts."""


def prefix_chunks(token_list: list[int], chunk: int = _PREFIX_CHUNK_TOKENS) -> list[str]:
    """Return ordered sha256 hex digests of non-overlapping token chunks.

    ``chunks[i]`` is the hash of tokens ``[i*chunk : (i+1)*chunk]``. This is
    the *key* used by peers to find a shared prefix: two instances that have
    prefilled the same first K tokens produce the same first ``K//chunk``
    chunk hashes, so a peer can answer "I hold this prefix" without sharing
    token arrays.
    """
    out: list[str] = []
    for i in range(0, len(token_list), chunk):
        block = token_list[i : i + chunk]
        # Tokens are ints (often >= 256); encode as stable text, not raw
        # bytes(block) which raises for values > 255.
        payload = ",".join(str(int(t)) for t in block).encode("utf-8")
        out.append(hashlib.sha256(payload).hexdigest())
    return out


@dataclass(frozen=True)
class PrefixDescriptor:
    """Compact, wire-serializable description of a cached prefix on one node."""

    model_hash: str
    """16-hex sha256 of the model_id (same scheme as cache.py disk dir)."""

    chunks: tuple[str, ...]
    """Ordered prefix chunk hashes (see :func:`prefix_chunks`)."""

    token_count: int
    """Total tokens in the cached prompt."""

    node_id: str
    """Owning node (system id)."""

    instance_id: str
    """Owning runner/instance id (disambiguates N runners per node)."""

    last_used: float = field(default_factory=time.time)
    """Monotonic-ish access time for LRU eviction of the *index* entry."""

    hits: int = 0
    """How many times this descriptor satisfied a peer lookup (local count)."""

    # -- small helpers ------------------------------------------------------

    @property
    def covered_tokens(self) -> int:
        return len(self.chunks) * _PREFIX_CHUNK_TOKENS

    def to_dict(self) -> dict[str, object]:
        return {
            "model_hash": self.model_hash,
            "chunks": list(self.chunks),
            "token_count": self.token_count,
            "node_id": self.node_id,
            "instance_id": self.instance_id,
            "last_used": self.last_used,
            "hits": self.hits,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PrefixDescriptor":
        return cls(
            model_hash=str(d["model_hash"]),
            chunks=tuple(str(c) for c in d["chunks"]),
            token_count=int(d["token_count"]),
            node_id=str(d["node_id"]),
            instance_id=str(d["instance_id"]),
            last_used=float(d["last_used"]),
            hits=int(d.get("hits", 0)),
        )


# ---------------------------------------------------------------------------
# Cluster prefix index (in-process, per-node)
# ---------------------------------------------------------------------------

_IndexKey: TypeAlias = tuple[str, tuple[str, ...]]
"""Index key: (model_hash, prefix_chunks)."""


class ClusterPrefixIndex:
    """Per-node advisory index of which peer/instance holds which prefix.

    Thread-safe (runners publish/consult from multiple threads). Entries are
    LRU-capped by :attr:`_max_entries`; stale entries (older than
    ``EXO_CLUSTER_PREFIX_CACHE_TTL``, default 300s) are pruned on access.

    This is intentionally *not* a radix tree: the findings doc (§3.2)
    establishes that a dict of token-chunk-hash tuples is functionally a
    compressed trie for hit-metrics purposes, with no pointer-chasing bugs.
    """

    def __init__(
        self,
        node_id: str = "unknown-node",
        max_entries: int | None = None,
        ttl: float | None = None,
    ) -> None:
        self.node_id = node_id
        self._max_entries = max_entries or int(
            os.environ.get("EXO_CLUSTER_PREFIX_CACHE_MAX_ENTRIES", "4096")
        )
        self._ttl = ttl if ttl is not None else float(
            os.environ.get("EXO_CLUSTER_PREFIX_CACHE_TTL", "300")
        )
        self._entries: dict[_IndexKey, PrefixDescriptor] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._published = 0

    # -- mutators -----------------------------------------------------------

    def clear(self) -> None:
        """Drop all entries (runner restart / node leave)."""
        with self._lock:
            self._entries.clear()

    def put_from_event(self, event: "PrefixIndexEvent") -> None:
        """Record an index entry from a network ``PrefixIndexEvent``."""
        self.put(
            PrefixDescriptor(
                model_hash=event.model_hash,
                chunks=tuple(event.chunks),
                token_count=event.token_count,
                node_id=event.node_id,
                instance_id=event.instance_id,
                last_used=event.last_used,
                hits=event.hits,
            )
        )

    def put(self, desc: PrefixDescriptor) -> None:
        """Record (or refresh) a descriptor from a peer or ourselves.

        The descriptor's *prefix expansion* is stored: for a descriptor with
        N chunks we store keys ``chunks[:1] .. chunks[:N]`` so a lookup of any
        shared prefix length finds it. This makes the dict a compressed trie
        (findings §3.2) at O(N) space per descriptor.
        """
        with self._lock:
            self._prune_locked()
            for i in range(1, len(desc.chunks) + 1):
                key = (desc.model_hash, desc.chunks[:i])
                existing = self._entries.get(key)
                if existing is not None:
                    # Refresh: bump LRU + hit count, keep token_count = the
                    # full covered length for the longest key.
                    self._entries[key] = PrefixDescriptor(
                        model_hash=desc.model_hash,
                        chunks=desc.chunks[:i],
                        token_count=min(desc.token_count, i * _PREFIX_CHUNK_TOKENS),
                        node_id=desc.node_id,
                        instance_id=desc.instance_id,
                        last_used=time.time(),
                        hits=existing.hits + 1,
                    )
                else:
                    if len(self._entries) >= self._max_entries:
                        self._evict_lru_locked()
                    self._entries[key] = PrefixDescriptor(
                        model_hash=desc.model_hash,
                        chunks=desc.chunks[:i],
                        token_count=min(desc.token_count, i * _PREFIX_CHUNK_TOKENS),
                        node_id=desc.node_id,
                        instance_id=desc.instance_id,
                        last_used=desc.last_used,
                        hits=0,
                    )

    def remove_node(self, node_id: str) -> int:
        """Drop all descriptors owned by ``node_id`` (node left / timed out)."""
        with self._lock:
            before = len(self._entries)
            self._entries = {
                k: v for k, v in self._entries.items() if v.node_id != node_id
            }
            return before - len(self._entries)

    # -- lookups ------------------------------------------------------------

    def lookup(
        self, model_hash: str, token_list: list[int], min_chunks: int = 1
    ) -> PrefixDescriptor | None:
        """Find a peer holding the longest shared prefix.

        ``min_chunks`` guards against trivial 1-chunk "hits" that would be
        cheaper to just prefill; callers should pass a floor derived from
        their measured prefill break-even.
        """
        chunks = prefix_chunks(token_list)
        if len(chunks) < min_chunks:
            return None
        with self._lock:
            self._prune_locked()
            best: PrefixDescriptor | None = None
            best_len = 0
            for prefix_len in range(len(chunks), 0, -1):
                key = (model_hash, tuple(chunks[:prefix_len]))
                desc = self._entries.get(key)
                if desc is not None and prefix_len > best_len:
                    best, best_len = desc, prefix_len
                    break  # longest-first scan; can stop at first hit
            if best is not None:
                self._hits += 1
                # refresh LRU
                self._entries[(best.model_hash, best.chunks)] = PrefixDescriptor(
                    model_hash=best.model_hash,
                    chunks=best.chunks,
                    token_count=best.token_count,
                    node_id=best.node_id,
                    instance_id=best.instance_id,
                    last_used=time.time(),
                    hits=best.hits + 1,
                )
            else:
                self._misses += 1
            return best

    # -- stats / maintenance -----------------------------------------------

    def descriptors(self) -> list[PrefixDescriptor]:
        """Return a snapshot of all live descriptors (collapsed to leaves).

        Prefix expansion inserts ``chunks[:i]`` for every ``i``, so one
        logical descriptor appears under every prefix length. Only the
        *deepest* key per ``(model_hash, full_chunks)`` is emitted; shorter
        keys are prefix views of the same descriptor and are dropped. Keeps
        the authoritative snapshot exactly one entry per logical descriptor.
        """
        with self._lock:
            # Build the maximal-key set: key K is a leaf iff no other live
            # key has the same model hash and a strictly longer chunk tuple
            # whose first len(K.chunks) entries equal K.chunks.
            all_keys = list(self._entries.keys())
            leaves: list[_IndexKey] = []
            for model_hash_, chunks in all_keys:
                is_leaf = True
                for other_model, other_chunks in all_keys:
                    if other_model != model_hash_:
                        continue
                    if (
                        len(other_chunks) > len(chunks)
                        and other_chunks[: len(chunks)] == chunks
                    ):
                        is_leaf = False
                        break
                if is_leaf:
                    leaves.append((model_hash_, chunks))
            seen: dict[_IndexKey, PrefixDescriptor] = {}
            for v in self._entries.values():
                key = (v.model_hash, v.chunks)
                if key not in leaves:
                    continue
                cur = seen.get(key)
                if cur is None or len(v.chunks) > len(cur.chunks):
                    seen[key] = v
            return list(seen.values())

    @property
    def stats(self) -> dict[str, object]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "hits": self._hits,
                "misses": self._misses,
                "published": self._published,
                "max_entries": self._max_entries,
                "ttl": self._ttl,
            }

    def _prune_locked(self) -> None:
        now = time.time()
        stale = [k for k, v in self._entries.items() if now - v.last_used > self._ttl]
        for k in stale:
            del self._entries[k]

    def _evict_lru_locked(self) -> None:
        if not self._entries:
            return
        lru_key = min(self._entries, key=lambda k: self._entries[k].last_used)
        del self._entries[lru_key]


# ---------------------------------------------------------------------------
# Topic/event wiring glue (pure, no mlx)
# ---------------------------------------------------------------------------

#: Env gate (default off -> zero behavior change).
ENV_CLUSTER_PREFIX_CACHE = "EXO_CLUSTER_PREFIX_CACHE"

#: Max entries in the worker-side authoritative index (env-tunable).
EXO_CLUSTER_PREFIX_INDEX_MAX_ENTRIES = int(
    os.environ.get("EXO_CLUSTER_PREFIX_INDEX_MAX_ENTRIES", "16384")
)

#: Topic name for prefix descriptors (consistent with the event-sourced plane).
PREFIX_INDEX_TOPIC = "prefix_index"


def cluster_prefix_cache_enabled() -> bool:
    return os.environ.get(ENV_CLUSTER_PREFIX_CACHE, "0") == "1"


def model_hash(model_id: str) -> str:
    """16-hex sha256 of model_id -- same scheme as cache.py ``_init_disk_dir``."""
    return hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# KVPrefixCache hook protocol
# ---------------------------------------------------------------------------


class ClusterPrefixHook:
    """Optional hook attached to :class:`KVPrefixCache`.

    The cache calls these at the three lifecycle points; implementations must
    be cheap and must never raise (the cache wraps calls in try/except and
    only logs warnings). All parameters are advisory descriptors -- never KV
    tensors.
    """

    def on_prefix_saved(
        self, model_id: str, prompt_tokens: "list[int] | Any"
    ) -> None:
        """Called after ``add_kv_cache`` / ``update_kv_cache``."""

    def on_prefix_miss(
        self,
        model_id: str,
        prompt_tokens: "list[int] | Any",
        media_regions: "list[Any] | None" = None,
    ) -> None:
        """Called when ``get_kv_cache`` finds no local entry."""

    def on_prefix_hit(
        self,
        model_id: str,
        prompt_tokens: "list[int] | Any",
        hit_length: int,
        is_exact: bool,
    ) -> None:
        """Called when ``get_kv_cache`` returns a local hit (partial or exact)."""


# ---------------------------------------------------------------------------
# Runner-side coordinator: owns the index + publishes/consumes peer events
# ---------------------------------------------------------------------------


class ClusterPrefixCoordinator(ClusterPrefixHook):
    """Wires a :class:`ClusterPrefixIndex` to the runner's event channel.

    Instantiated by the runner when ``EXO_CLUSTER_PREFIX_CACHE=1`` and passed
    to ``KVPrefixCache(cluster_hook=...)``. On save it publishes a compact
    :class:`PrefixIndexEvent` over ``event_sender`` (the same channel that
    carries all runner events); on miss/hit it consults the local index and
    logs ``PREFIX_CLUSTER_HIT`` / ``PREFIX_CLUSTER_MISS`` events so the
    cluster-level reuse is observable.

    Never raises: all callbacks are wrapped; a broken event channel or
    index degrades silently (the per-instance cache path is untouched).
    """

    def __init__(
        self,
        node_id: str,
        instance_id: str,
        event_sender: "Any | None" = None,
        min_chunks: int = 1,
        index: "ClusterPrefixIndex | None" = None,
    ) -> None:
        self.node_id = node_id
        self.instance_id = instance_id
        self.event_sender = event_sender
        self.min_chunks = min_chunks
        self.index = index if index is not None else ClusterPrefixIndex(node_id=node_id)
        self._model_hash_cache: dict[str, str] = {}

    # -- ClusterPrefixHook impl --------------------------------------------

    def on_prefix_saved(
        self, model_id: str, prompt_tokens: "list[int] | Any"
    ) -> None:
        try:
            tokens = _as_token_list(prompt_tokens)
            if len(tokens) < 1:
                return
            chunks = prefix_chunks(tokens)
            if not chunks:
                return
            self._publish_saved(model_id, chunks, token_count=len(tokens))
        except Exception:
            logger.warning("Cluster prefix publish failed", exc_info=True)

    def on_prefix_miss(
        self,
        model_id: str,
        prompt_tokens: "list[int] | Any",
        media_regions: "list[Any] | None" = None,
    ) -> None:
        try:
            tokens = _as_token_list(prompt_tokens)
            mh = self._model_hash(model_id)
            hit = self.index.lookup(mh, tokens, min_chunks=self.min_chunks)
            if hit is not None:
                self._emit_event("PREFIX_CLUSTER_HIT", hit=hit)
            else:
                self._emit_event("PREFIX_CLUSTER_MISS")
        except Exception:
            logger.warning("Cluster prefix lookup failed", exc_info=True)

    def on_prefix_hit(
        self,
        model_id: str,
        prompt_tokens: "list[int] | Any",
        hit_length: int,
        is_exact: bool,
    ) -> None:
        # A local hit means the cluster index is satisfied by this node;
        # nothing to query. We still record the local reuse for stats.
        with contextlib.suppress(Exception):
            self._emit_event("PREFIX_LOCAL_HIT", hit_length=hit_length)

    # -- peer event consumption --------------------------------------------

    def consume(self, event: "Any") -> None:
        """Apply a peer's ``PrefixIndexEvent`` to the local index."""
        try:
            # Reject our own echoes: the event stream loops back to the
            # publisher, and re-adding our own descriptors is a no-op that
            # would just refresh LRU. Filtering keeps the index peer-only.
            if getattr(event, "node_id", None) == self.node_id:
                return
            desc = PrefixDescriptor(
                model_hash=event.model_hash,
                chunks=tuple(event.chunks),
                token_count=event.token_count,
                node_id=event.node_id,
                instance_id=event.instance_id,
                last_used=event.last_used,
                hits=event.hits,
            )
            self.index.put(desc)
        except Exception:
            logger.warning("Cluster prefix consume failed", exc_info=True)

    def drop_node(self, node_id: str) -> int:
        return self.index.remove_node(node_id)

    def lookup_peer(
        self, model_id: str, token_list: "list[int] | Any", min_chunks: int | None = None
    ) -> "PrefixDescriptor | None":
        """Advisory: which peer holds the longest shared prefix of ``token_list``?"""
        mh = self._model_hash(model_id)
        tokens = _as_token_list(token_list)
        return self.index.lookup(mh, tokens, min_chunks=min_chunks or self.min_chunks)

    # -- internals ----------------------------------------------------------

    def _publish_saved(
        self, model_id: str, chunks: list[str], token_count: int
    ) -> None:
        if self.event_sender is None:
            return
        from exo.shared.types.events import PrefixIndexEvent

        ev = PrefixIndexEvent(
            model_hash=self._model_hash(model_id),
            chunks=tuple(chunks),
            token_count=token_count,
            node_id=self.node_id,
            instance_id=self.instance_id,
            last_used=time.time(),
            hits=0,
        )
        try:
            self.event_sender.send(ev)
            self.index._published += 1
        except Exception:
            logger.warning("Cluster prefix event send failed", exc_info=True)

    def _model_hash(self, model_id: str) -> str:
        if model_id not in self._model_hash_cache:
            self._model_hash_cache[model_id] = model_hash(model_id)
        return self._model_hash_cache[model_id]

    def _emit_event(self, kind: str, **fields: object) -> None:
        if self.event_sender is None:
            return
        # Local observable marker: the index events flow through the same
        # event plane; wire-level stats events are deliberately not added
        # in v1 (avoid spamming the cluster bus with per-request events).
        logger.info(f"Cluster prefix [{kind}] {fields}")


def _as_token_list(prompt_tokens: "Any") -> list[int]:
    """Best-effort conversion of mx.array / list / numpy to a flat int list."""
    if prompt_tokens is None:
        return []
    # mx.array exposes .tolist(); numpy arrays too.
    tolist = getattr(prompt_tokens, "tolist", None)
    if callable(tolist):
        raw: Any = tolist()
    else:
        raw = list(prompt_tokens)  # type: ignore[arg-type]
    out: list[int] = []
    for x in raw:
        if isinstance(x, (list, tuple)):
            out.extend(int(v) for v in x)
        else:
            out.append(int(x))
    return out