"""Thread-safe per-request latency tracker for the API layer.

Tracks TTFT (time to first token), total wall time, and overall tokens/sec
at the API level.  Engine-level decode latency is populated by the generator
itself and merged here.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from exo.api.types import GenerationStats
from exo.shared.types.common import CommandId


@dataclass
class _RequestTiming:
    request_started: float
    first_token_received: float | None = None


class RequestLatencyTracker:
    def __init__(self) -> None:
        self._timings: dict[CommandId, _RequestTiming] = {}
        self._lock = threading.Lock()

    def record_request(self, command_id: CommandId) -> None:
        """Mark the moment a text generation command was submitted."""
        with self._lock:
            self._timings[command_id] = _RequestTiming(request_started=time.monotonic())

    def record_first_token(self, command_id: CommandId) -> float | None:
        """Record first-token timestamp; return TTFT in ms (or None)."""
        now = time.monotonic()
        with self._lock:
            t = self._timings.get(command_id)
            if t is None:
                return None
            if t.first_token_received is None:
                t.first_token_received = now
                return (now - t.request_started) * 1000
            # Already recorded — return the cached value.
            return (t.first_token_received - t.request_started) * 1000

    def compute_total_time_ms(self, command_id: CommandId) -> float | None:
        """Compute total wall time and remove tracking entry."""
        now = time.monotonic()
        with self._lock:
            t = self._timings.pop(command_id, None)
            if t is None:
                return None
            return (now - t.request_started) * 1000

    def merge_api_stats(
        self, command_id: CommandId, stats: GenerationStats | None
    ) -> GenerationStats | None:
        """Merge API-level timing stats into engine-provided GenerationStats."""
        ttft_ms = self.record_first_token(command_id)
        total_ms = self.compute_total_time_ms(command_id)
        if stats is None:
            return stats
        update: dict[str, object] = {}
        if ttft_ms is not None and stats.time_to_first_token_ms is None:
            update["time_to_first_token_ms"] = ttft_ms
        if total_ms is not None and stats.total_time_ms is None:
            update["total_time_ms"] = total_ms
        # Compute overall tokens/sec if we have both data points.
        if update and total_ms is not None and total_ms > 0 and stats.generation_tokens > 0:
            update.setdefault(
                "tokens_per_second",
                stats.generation_tokens / (total_ms / 1000),
            )
        if update:
            return stats.model_copy(update=update)
        return stats


class _RequestTimingState:
    """Helper attached to a single streaming response to enrich the final chunk."""

    def __init__(self, tracker: RequestLatencyTracker, command_id: CommandId) -> None:
        self._tracker = tracker
        self._command_id = command_id
        self._ttft_ms: float | None = None

    def process_chunk(self, chunk: object) -> object:
        """Maybe enrich *chunk* with API-level timing and return it (or a copy)."""
        from exo.shared.types.chunks import PrefillProgressChunk, ToolCallChunk, TokenChunk

        if isinstance(chunk, PrefillProgressChunk):
            return chunk
        if not isinstance(chunk, (TokenChunk, ToolCallChunk)):
            return chunk

        # Record first-token timing on the first real token.
        if self._ttft_ms is None:
            self._ttft_ms = self._tracker.record_first_token(self._command_id)

        # Enrich the final chunk (has finish_reason) with timing stats.
        if (
            chunk.finish_reason is not None
            and chunk.stats is not None
            and self._ttft_ms is not None
        ):
            total_ms = self._tracker.compute_total_time_ms(self._command_id)
            gen_tokens = chunk.stats.generation_tokens or 0
            tps = (gen_tokens / (total_ms / 1000)) if total_ms and total_ms > 0 else None
            return chunk.model_copy(
                update={
                    "stats": chunk.stats.model_copy(
                        update={
                            "time_to_first_token_ms": self._ttft_ms,
                            "total_time_ms": total_ms,
                            "tokens_per_second": tps,
                        }
                    )
                }
            )
        return chunk
