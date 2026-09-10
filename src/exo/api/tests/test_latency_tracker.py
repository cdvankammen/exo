"""Tests for per-request latency tracking (RequestLatencyTracker + stats merge)."""

import time

from exo.api.latency_tracker import RequestLatencyTracker, _RequestTimingState
from exo.api.types import GenerationStats
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.memory import Memory


def _stats(**overrides) -> GenerationStats:
    values = dict(
        prompt_tps=100.0,
        generation_tps=50.0,
        prompt_tokens=10,
        generation_tokens=5,
        peak_memory_usage=Memory(in_bytes=1024),
    )
    values.update(overrides)
    return GenerationStats(**values)


def test_record_request_and_first_token() -> None:
    tracker = RequestLatencyTracker()
    cid = CommandId("test-1")
    tracker.record_request(cid)
    time.sleep(0.01)
    ttft = tracker.record_first_token(cid)
    assert ttft is not None
    assert ttft > 0

    # Second call returns the cached value, doesn't re-timestamp
    ttft2 = tracker.record_first_token(cid)
    assert ttft2 == ttft


def test_compute_total_time_removes_entry() -> None:
    tracker = RequestLatencyTracker()
    cid = CommandId("test-2")
    tracker.record_request(cid)
    time.sleep(0.01)
    total = tracker.compute_total_time_ms(cid)
    assert total is not None
    assert total > 0
    # Entry is popped after compute
    assert tracker.compute_total_time_ms(cid) is None


def test_record_first_token_without_request_returns_none() -> None:
    tracker = RequestLatencyTracker()
    assert tracker.record_first_token(CommandId("ghost")) is None
    assert tracker.compute_total_time_ms(CommandId("ghost")) is None


def test_merge_api_stats_populates_timing() -> None:
    tracker = RequestLatencyTracker()
    cid = CommandId("test-3")
    tracker.record_request(cid)
    time.sleep(0.01)
    merged = tracker.merge_api_stats(cid, _stats())
    assert merged is not None
    assert merged.time_to_first_token_ms is not None
    assert merged.time_to_first_token_ms > 0
    assert merged.total_time_ms is not None
    assert merged.total_time_ms > 0
    assert merged.tokens_per_second is not None
    assert merged.tokens_per_second > 0
    # Engine-populated fields survive the merge
    assert merged.prompt_tps == 100.0
    assert merged.generation_tps == 50.0


def test_merge_api_stats_keeps_engine_decode_latency() -> None:
    tracker = RequestLatencyTracker()
    cid = CommandId("test-4")
    tracker.record_request(cid)
    merged = tracker.merge_api_stats(cid, _stats(decode_latency_ms=123.0))
    assert merged is not None
    assert merged.decode_latency_ms == 123.0
    assert merged.time_to_first_token_ms is not None


def test_merge_api_stats_with_none_stats() -> None:
    tracker = RequestLatencyTracker()
    cid = CommandId("test-5")
    tracker.record_request(cid)
    assert tracker.merge_api_stats(cid, None) is None


def test_timing_state_enriches_final_chunk() -> None:
    tracker = RequestLatencyTracker()
    cid = CommandId("test-6")
    tracker.record_request(cid)

    state = _RequestTimingState(tracker, cid)

    # Prefill chunk passes through untouched
    from exo.shared.types.chunks import PrefillProgressChunk

    prefill = PrefillProgressChunk(model=ModelId("m"), processed_tokens=1, total_tokens=5)
    assert state.process_chunk(prefill) is prefill

    # First token chunk: recorded, no enrichment yet
    first = TokenChunk(
        model=ModelId("m"),
        token_id=1,
        text="Hello",
        usage=None,
    )
    enriched = state.process_chunk(first)
    assert enriched is first  # no stats yet to merge

    # Final chunk with stats: enriched with API timing
    final = TokenChunk(
        model=ModelId("m"),
        token_id=2,
        text=" world",
        usage=None,
        finish_reason="stop",
        stats=_stats(),
    )
    enriched_final = state.process_chunk(final)
    assert enriched_final is not final  # copied
    assert enriched_final.stats is not None
    assert enriched_final.stats.time_to_first_token_ms is not None
    assert enriched_final.stats.time_to_first_token_ms > 0
    assert enriched_final.stats.total_time_ms is not None
    assert enriched_final.stats.tokens_per_second is not None