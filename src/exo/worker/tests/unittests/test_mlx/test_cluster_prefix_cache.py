"""Unit tests for the cluster-wide prefix cache (cross-node coordination).

Coverage:
- ``ClusterPrefixIndex``: put / lookup / nearest-prefix / prefix expansion /
  dedup / LRU capacity / descriptors snapshot / clear / TTL.
- ``ClusterPrefixCoordinator``: on_prefix_saved publishes an event;
  consume() absorbs peer events; own-event rejection; lookup_peer.
- ``prefix_chunks`` / ``model_hash`` utilities.
- Discriminated-union membership: ``PrefixIndexEvent`` is an ``Event``,
  ``PrefixIndexSnapshotTask`` is a ``Task``.
"""

from __future__ import annotations

import hashlib
import os
import time
from unittest.mock import MagicMock

import pytest

from exo.shared.types.events import Event, PrefixIndexEvent
from exo.shared.types.tasks import PrefixIndexSnapshotTask, Task
from exo.worker.engines.mlx.cluster_cache import (
    ENV_CLUSTER_PREFIX_CACHE,
    _PREFIX_CHUNK_TOKENS,
    ClusterPrefixCoordinator,
    ClusterPrefixIndex,
    PrefixDescriptor,
    cluster_prefix_cache_enabled,
    model_hash,
    prefix_chunks,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _tokens(*blocks: list[int]) -> list[int]:
    """Build a token list whose 256-token chunks equal the given blocks.

    Real ``prefix_chunks`` groups tokens in ``_PREFIX_CHUNK_TOKENS`` (256)
    blocks. Test descriptors must be built from token lists of exactly
    256 tokens per chunk so that ``prefix_chunks(lookup_tokens)`` produces
    the same chunk hashes stored by ``_desc``. This helper pads each block
    to 256 tokens with distinguishable filler.
    """
    out: list[int] = []
    for bi, block in enumerate(blocks):
        block = list(block)
        if len(block) < _PREFIX_CHUNK_TOKENS:
            filler = _PREFIX_CHUNK_TOKENS - len(block)
            # Use the block index as a filler base so different blocks never
            # collide with each other (or with the real tokens).
            block = block + [100_000 + bi * _PREFIX_CHUNK_TOKENS + i for i in range(filler)]
        assert len(block) == _PREFIX_CHUNK_TOKENS
        out.extend(block)
    return out


def _h(*token_blocks: list[int]) -> tuple[str, ...]:
    """Compute chunk hashes exactly like ``prefix_chunks`` for the given
    256-token-aligned blocks (see ``_tokens``)."""
    return tuple(prefix_chunks(_tokens(*token_blocks)))


def _desc(
    *,
    model: str = "m",
    chunks: tuple[str, ...] = _h([11], [22], [33]),
    count: int = 300,
    node: str = "n1",
    instance: str = "i1",
    hits: int = 0,
    last_used: float | None = None,
) -> PrefixDescriptor:
    return PrefixDescriptor(
        model_hash=model,
        chunks=chunks,
        token_count=count,
        node_id=node,
        instance_id=instance,
        last_used=last_used if last_used is not None else time.time(),
        hits=hits,
    )


# ── primitives ───────────────────────────────────────────────────────────

class TestPrefixChunks:
    def test_small_prompt(self) -> None:
        chunks = prefix_chunks(list(range(10)))
        assert len(chunks) == 1
        assert chunks == prefix_chunks(list(range(10)))  # deterministic

    def test_large_prompt_splits(self) -> None:
        tokens = list(range(3 * _PREFIX_CHUNK_TOKENS + 5))
        chunks = prefix_chunks(tokens)
        assert len(chunks) == 4

    def test_longest_common_prefix(self) -> None:
        a = prefix_chunks(list(range(1000)))
        b = prefix_chunks(list(range(900)))
        overlap = 0
        for x, y in zip(a, b):
            if x != y:
                break
            overlap += 1
        assert overlap == 3

    def test_empty(self) -> None:
        assert prefix_chunks([]) == []

    def test_matches_helper(self) -> None:
        """Verify _h helper produces identical hashes to prefix_chunks."""
        tokens = _tokens([10, 20, 30])
        assert _h([10, 20, 30]) == tuple(prefix_chunks(tokens))


class TestModelHash:
    def test_deterministic_and_short(self) -> None:
        h = model_hash("mlx-community/Llama-3.2-1B-Instruct-4bit")
        assert len(h) == 16
        assert h == model_hash("mlx-community/Llama-3.2-1B-Instruct-4bit")

    def test_differs_per_model(self) -> None:
        assert model_hash("a") != model_hash("b")


# ── index ────────────────────────────────────────────────────────────────

class TestClusterPrefixIndex:
    def test_put_lookup_exact(self) -> None:
        idx = ClusterPrefixIndex(node_id="n1", max_entries=100)
        idx.put(_desc(chunks=_h([11], [22], [33]), node="n2", instance="i2"))
        hit = idx.lookup("m", _tokens([11], [22], [33]))
        assert hit is not None
        assert hit.node_id == "n2"
        # descriptor's token_count reflects the full covered prompt length
        assert hit.token_count == 300

    def test_put_lookup_partial(self) -> None:
        idx = ClusterPrefixIndex(node_id="n1", max_entries=100)
        idx.put(_desc(chunks=_h([11], [22], [33]), node="n2", instance="i2"))
        # Query with a longer prompt: first 2 blocks match.
        hit = idx.lookup("m", _tokens([11], [22], [99]))
        assert hit is not None
        assert len(hit.chunks) == 2  # prefix length in chunks

    def test_unknown_model_misses(self) -> None:
        idx = ClusterPrefixIndex(node_id="n1", max_entries=100)
        idx.put(_desc(chunks=_h([11])))
        assert idx.lookup("other-model", _tokens([11])) is None

    def test_own_node_stored_by_index(self) -> None:
        """The index stores any descriptor (incl. own-node); the consume()
        guard is what rejects echoes of our own events."""
        idx = ClusterPrefixIndex(node_id="me", max_entries=100)
        idx.put(_desc(chunks=_h([11]), node="me", instance="i1"))
        hit = idx.lookup("m", _tokens([11]))
        assert hit is not None
        assert hit.node_id == "me"

    def test_prefix_expansion_dedup(self) -> None:
        idx = ClusterPrefixIndex(node_id="n1", max_entries=100)
        c = _h([11], [22], [33])
        idx.put(_desc(chunks=c, node="n2", instance="i2"))
        idx.put(_desc(chunks=c, node="n2", instance="i2"))  # dup
        # 3 prefix-length keys stored; duplicate put refreshes, does not grow
        assert idx.stats["entries"] == 3
        # longest dedups to a single descriptor
        assert len(idx.descriptors()) == 1

    def test_capacity_evicts_lru(self) -> None:
        idx = ClusterPrefixIndex(node_id="n1", max_entries=4)
        idx.put(_desc(chunks=_h([11]), node="n2", instance="i1"))
        idx.put(_desc(chunks=_h([22]), node="n2", instance="i2"))
        idx.put(_desc(chunks=_h([33]), node="n3", instance="i3"))
        idx.put(_desc(chunks=_h([44]), node="n3", instance="i4"))
        assert len(idx.descriptors()) == 4
        # touch 11 so it is most recent
        idx.lookup("m", _tokens([11]))
        idx.put(_desc(chunks=_h([55]), node="n4", instance="i5"))
        descs = {d.chunks[0]: d for d in idx.descriptors()}
        # 22 was never touched -> evicted
        assert _h([22])[0] not in descs
        assert _h([55])[0] in descs

    def test_descriptors_snapshot(self) -> None:
        idx = ClusterPrefixIndex(node_id="n1", max_entries=100)
        idx.put(_desc(chunks=_h([11]), node="n2", instance="i1"))
        idx.put(_desc(chunks=_h([11], [22]), node="n2", instance="i1"))
        assert len(idx.descriptors()) == 2

    def test_clear(self) -> None:
        idx = ClusterPrefixIndex(node_id="n1", max_entries=100)
        idx.put(_desc(chunks=_h([11])))
        idx.clear()
        assert idx.descriptors() == []
        assert idx.lookup("m", _tokens([11])) is None

    def test_ttl_eviction(self) -> None:
        idx = ClusterPrefixIndex(node_id="n1", max_entries=100, ttl=0.01)
        idx.put(_desc(chunks=_h([11]), node="n2", instance="i2"))
        time.sleep(0.02)
        assert idx.lookup("m", _tokens([11])) is None


# ── coordinator ──────────────────────────────────────────────────────────

class TestClusterPrefixCoordinator:
    def test_gated_off_by_default(self) -> None:
        os.environ.pop(ENV_CLUSTER_PREFIX_CACHE, None)
        assert not cluster_prefix_cache_enabled()

    def test_gated_on_by_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_CLUSTER_PREFIX_CACHE, "1")
        assert cluster_prefix_cache_enabled()

    def test_on_prefix_saved_publishes_event(self) -> None:
        sender = MagicMock()
        coord = ClusterPrefixCoordinator(
            node_id="n1",
            instance_id="i1",
            event_sender=sender,
            index=ClusterPrefixIndex(node_id="n1", max_entries=100),
        )
        coord.on_prefix_saved("model-x", list(range(100)))
        assert sender.send.call_count == 1
        event = sender.send.call_args.args[0]
        assert isinstance(event, PrefixIndexEvent)
        assert event.node_id == "n1"
        assert event.instance_id == "i1"
        assert event.model_hash == model_hash("model-x")
        assert len(event.chunks) == 1

    def test_consume_absorbs_peer(self) -> None:
        idx = ClusterPrefixIndex(node_id="n1", max_entries=100)
        chunks = _h([11], [22])
        coord = ClusterPrefixCoordinator(
            node_id="n1",
            instance_id="i1",
            event_sender=None,
            index=idx,
        )
        coord.consume(
            PrefixIndexEvent(
                model_hash="m",
                chunks=chunks,
                token_count=512,
                node_id="n2",
                instance_id="i2",
                last_used=time.time(),
            )
        )
        hit = idx.lookup("m", _tokens([11], [22]))
        assert hit is not None
        assert hit.node_id == "n2"

    def test_consume_ignores_own_events(self) -> None:
        idx = ClusterPrefixIndex(node_id="n1", max_entries=100)
        coord = ClusterPrefixCoordinator(
            node_id="n1", instance_id="i1", event_sender=None, index=idx
        )
        coord.consume(
            PrefixIndexEvent(
                model_hash="m",
                chunks=_h([11]),
                token_count=256,
                node_id="n1",  # own node -> reject
                instance_id="i1",
                last_used=time.time(),
            )
        )
        assert idx.descriptors() == []

    def test_lookup_returns_peer_hint(self) -> None:
        idx = ClusterPrefixIndex(node_id="n1", max_entries=100)
        # The coordinator hashes model ids internally; store under the hash.
        idx.put(_desc(model=model_hash("m"), chunks=_h([11], [22]), node="n2", instance="i2"))
        coord = ClusterPrefixCoordinator(
            node_id="n1", instance_id="i1", event_sender=None, index=idx
        )
        hint = coord.lookup_peer("m", _tokens([11], [33]))
        assert hint is not None
        assert hint.node_id == "n2"

    def test_lookup_peer_returns_none_for_miss(self) -> None:
        idx = ClusterPrefixIndex(node_id="n1", max_entries=100)
        coord = ClusterPrefixCoordinator(
            node_id="n1", instance_id="i1", event_sender=None, index=idx
        )
        assert coord.lookup_peer("m", _tokens([99])) is None


# ── union membership ─────────────────────────────────────────────────────

class TestUnionMembership:
    def test_prefix_index_event_is_event(self) -> None:
        ev = PrefixIndexEvent(
            model_hash="m",
            chunks=_h([11]),
            token_count=256,
            node_id="n2",
            instance_id="i2",
            last_used=time.time(),
        )
        assert isinstance(ev, Event)

    def test_snapshot_task_is_task(self) -> None:
        t = PrefixIndexSnapshotTask(instance_id="i1", descriptors=[])
        assert isinstance(t, Task)

    def test_snapshot_task_roundtrips_descriptors(self) -> None:
        desc = _desc(chunks=_h([11], [22]), node="n2", instance="i2")
        d = PrefixDescriptor.to_dict(desc)
        restored = PrefixDescriptor.from_dict(d)
        assert restored.node_id == desc.node_id
        assert restored.chunks == desc.chunks

    def test_event_serializes(self) -> None:
        """PrefixIndexEvent round-trips through Pydantic serialization."""
        ev = PrefixIndexEvent(
            model_hash="m",
            chunks=_h([11], [22], [33]),
            token_count=768,
            node_id="n1",
            instance_id="i1",
            last_used=time.time(),
            hits=3,
        )
        rt = PrefixIndexEvent.model_validate(ev.model_dump())
        assert rt.chunks == ev.chunks
        assert rt.hits == 3
