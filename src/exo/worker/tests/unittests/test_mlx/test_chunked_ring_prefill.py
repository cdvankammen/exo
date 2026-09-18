# type: ignore
"""Tests for chunked ring prefill (interleaved with decode).

Covers the pure chunk-boundary helpers and the chunked ``RingPrefillState``
that lets ``ExoBatchGenerator`` drain ring prefill one chunk per decode step
instead of blocking ``submit()`` on the whole prompt.

The helpers are pure arithmetic (no distributed group needed).  The state
test uses a ``FakeGroup`` and monkeypatched collectives, mirroring
``test_ring_attention.py`` conventions.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from exo.worker.engines.mlx.generator.generate import (
    RingPrefillState,
    compute_ring_prefill_chunks,
    compute_ring_prefill_rank_bounds,
    make_ring_prefill_state,
)
from exo.worker.engines.mlx.ring_attention import RingAttentionLayer


# ── compute_ring_prefill_chunks ─────────────────────────────────────────── #


class TestComputeRingPrefillChunks:
    def test_single_chunk_when_small(self) -> None:
        chunks = compute_ring_prefill_chunks(num_tokens=10, world_size=4, max_chunk_size=16)
        assert chunks == [(0, 10)]

    def test_exact_multiples(self) -> None:
        chunks = compute_ring_prefill_chunks(num_tokens=64, world_size=4, max_chunk_size=16)
        assert chunks == [(0, 16), (16, 32), (32, 48), (48, 64)]

    def test_trailing_partial_merged_into_previous(self) -> None:
        # 98 = 96 + 2; trailing 2 < world_size 4 → merge into chunk (80, 96).
        chunks = compute_ring_prefill_chunks(num_tokens=98, world_size=4, max_chunk_size=16)
        assert chunks == [
            (0, 16),
            (16, 32),
            (32, 48),
            (48, 64),
            (64, 80),
            (80, 98),
        ]

    def test_trailing_exact_world_size_kept(self) -> None:
        # 100 = 96 + 4; trailing exactly world_size → kept as its own chunk.
        chunks = compute_ring_prefill_chunks(num_tokens=100, world_size=4, max_chunk_size=16)
        assert chunks[-1] == (96, 100)
        assert len(chunks) == 7

    def test_no_chunks_for_empty(self) -> None:
        assert compute_ring_prefill_chunks(num_tokens=0, world_size=2, max_chunk_size=16) == []

    def test_contract_chunks_at_least_world_size(self) -> None:
        # Any non-final chunk must be >= world_size so every rank gets a
        # non-empty slice (contract relied on by the ring collective).
        for num_tokens in (16, 17, 33, 64, 98, 100, 1000):
            for ws in (2, 4, 8):
                chunks = compute_ring_prefill_chunks(
                    num_tokens, ws, max_chunk_size=ws * 16
                )
                for cs, ce in chunks[:-1]:
                    assert ce - cs >= ws, f"chunk {(cs, ce)} < world_size {ws}"


# ── compute_ring_prefill_rank_bounds ────────────────────────────────────── #


class TestComputeRingPrefillRankBounds:
    def test_even_split(self) -> None:
        # 16 tokens, 4 ranks → 4 tokens each.
        assert compute_ring_prefill_rank_bounds(0, 16, 0, 4) == (0, 4)
        assert compute_ring_prefill_rank_bounds(0, 16, 1, 4) == (4, 8)
        assert compute_ring_prefill_rank_bounds(0, 16, 2, 4) == (8, 12)
        assert compute_ring_prefill_rank_bounds(0, 16, 3, 4) == (12, 16)

    def test_floor_division_contiguous_cover(self) -> None:
        # 10 tokens, 4 ranks: floor splits (0,2),(2,5),(5,7),(7,10) —
        # contiguous, non-empty, exact coverage of [0, 10).
        bounds = [compute_ring_prefill_rank_bounds(0, 10, r, 4) for r in range(4)]
        assert bounds[0] == (0, 2)
        assert bounds[1] == (2, 5)
        assert bounds[2] == (5, 7)
        assert bounds[3] == (7, 10)
        assert bounds[0][0] == 0 and bounds[-1][1] == 10
        for (_, e), (s, _) in zip(bounds, bounds[1:]):
            assert e == s  # contiguous

    def test_rank_bounds_cover_chunk(self) -> None:
        for cs, ce in [(0, 16), (16, 32), (80, 98), (96, 100)]:
            for ws in (2, 4, 8):
                parts = [compute_ring_prefill_rank_bounds(cs, ce, r, ws) for r in range(ws)]
                # Exact coverage, no gaps / overlaps.
                assert parts[0][0] == cs
                assert parts[-1][1] == ce
                for (_, e), (s, _) in zip(parts, parts[1:]):
                    assert e == s


# ── RingPrefillState (fake group, fake model) ────────────────────────────── #


class FakeRankGroup:
    """Minimal distributed group standing in for ``mx.distributed.Group``."""

    def __init__(self, rank: int, size: int) -> None:
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


class _InnerModel(nn.Module):
    """Inner ``model`` attribute so ``ring_prefill_block_size`` can traverse.

    Provides a single ring-attention layer with ``sequence_block_size=4``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers: list[nn.Module] = [_RingLayer()]


class _RingLayer(RingAttentionLayer):
    """Minimal RingAttentionLayer exposing ``sequence_block_size``."""

    def __init__(self) -> None:
        super().__init__(nn.Linear(4, 4), FakeRankGroup(0, 2))
        self.sequence_block_size = 4


class FakeLM(nn.Module):
    """A model whose forward records (prompt slice span, cache) per call."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[int, int, object]] = []
        # ring_prefill_block_size needs a real nn.Module inner model with
        # ring-attention layers.
        self.model = _InnerModel()

    def __call__(self, x: mx.array, cache: object | None = None, **kwargs: object) -> mx.array:
        # x is [1, seq] — record the slice span of this chunk call.
        seq = x.shape[1]
        self.calls.append((seq, x.shape[0], cache))
        return mx.zeros((1, seq, 4), dtype=mx.float16)


def _fake_cache(n_layers: int = 2) -> list[object]:
    class _C:
        def __init__(self) -> None:
            self.state = mx.zeros((1, 1, 0, 1))

    return [_C() for _ in range(n_layers)]


class TestRingPrefillState:
    def test_steps_one_chunk_per_call_and_exhausts(self) -> None:
        model = FakeLM()
        cache = _fake_cache()
        state = RingPrefillState(
            model=model,
            prompt_tokens=mx.arange(32),
            cache=cache,
            group=FakeRankGroup(rank=0, size=2),
            chunk_ranges=[(0, 16), (16, 32)],
            rank=0,
            world_size=2,
        )
        assert state.step() is True  # chunk 0
        assert state.index == 1
        assert state.step() is True  # chunk 1
        assert state.index == 2
        assert state.step() is False  # exhausted
        assert len(model.calls) == 2
        # Each chunk call was a [1, seq] slice.
        for seq, batch, _ in model.calls:
            assert batch == 1
            assert seq == 8  # 16 tokens / 2 ranks

    def test_rank_slices_divide_chunk(self) -> None:
        model = FakeLM()
        state = RingPrefillState(
            model=model,
            prompt_tokens=mx.arange(32),
            cache=_fake_cache(),
            group=FakeRankGroup(rank=1, size=2),
            chunk_ranges=[(0, 16), (16, 32)],
            rank=1,
            world_size=2,
        )
        state.step()
        state.step()
        # Rank 1 processes the second half of each chunk: tokens 8..16, 24..32.
        assert model.calls[0][0] == 8
        assert model.calls[1][0] == 8

    def test_make_ring_prefill_state_chunking(self) -> None:
        # 100 tokens, world_size 4, block_size 4 → chunk size 16.
        state = make_ring_prefill_state(
            FakeLM(), mx.arange(100), _fake_cache(), FakeRankGroup(rank=0, size=4)
        )
        # 100 = 6 * 16 + 4; trailing 4 == world_size → kept. 7 chunks.
        assert [ce - cs for cs, ce in state.chunk_ranges] == [16] * 6 + [4]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))