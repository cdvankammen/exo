"""Chunked ring prefill state machine for interleaving with decode.

Ring prefill is all-or-nothing: the full ring-attention loop in ``prefill()``
runs synchronously, blocking decode for every other request in the continuous
batch.  This module splits that loop into discrete chunks that can be advanced
one at a time by the batch scheduler, allowing decode steps to run between
prefill chunks and improving latency for concurrent requests.

The :class:`ChunkedRingPrefill` object is created by ``ExoBatchGenerator``
when a ring-eligible prompt arrives.  The batch scheduler calls
``advance_chunk`` once per scheduler tick (one chunk per tick, so a decode
step can run between chunks), and calls ``finish`` once all chunks complete
to transition to decode.

All ranks advance the same chunk each tick — the scheduler keeps them in
lockstep — so the ring send/recv inside each chunk stays synchronized.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable

import mlx.core as mx
from mlx_lm.generate import generation_stream

from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.utils.stall_watchdog import StallWatchdog
from exo.worker.engines.mlx.cache import CacheSnapshot
from exo.worker.engines.mlx.ring_attention import (
    ring_prefill_block_size,
    set_ring_prefill,
)
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.engines.mlx.vision import MediaRegion, VisionResult
from exo.worker.runner.bootstrap import logger

# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
@dataclass
class ChunkedRingPrefill:
    """State machine that advances one ring prefill chunk per call.

    Created by ``ExoBatchGenerator.submit()`` for ring-eligible prompts.
    The batch generator's ``step()`` calls ``advance_chunk()`` once per
    scheduler tick, processing exactly one chunk of the ring prefill and
    returning control to the scheduler so decode can run between chunks.
    """

    prompt_tokens: mx.array  # full prompt minus the last token
    cache: KVCacheType
    model: Model
    group: mx.distributed.Group
    prefix_hit_length: int
    on_prefill_progress: Callable[[int, int], None] | None
    distributed_prompt_progress_callback: Callable[[], None] | None

    # Internals — set by __post_init__
    _rank: int = field(init=False, repr=False)
    _world_size: int = field(init=False, repr=False)
    _max_chunk_size: int = field(init=False, repr=False)
    _total_tokens: int = field(init=False, repr=False)
    _chunk_ranges: list[int] = field(init=False, repr=False)
    _chunk_index: int = field(init=False, repr=False, default=0)
    _all_done: bool = field(init=False, repr=False, default=False)
    _done_at: float = field(init=False, repr=False, default=0.0)
    _t_start: float = field(init=False, repr=False, default=0.0)
    _watchdog: StallWatchdog | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._rank = self.group.rank()
        self._world_size = self.group.size()
        self._total_tokens = len(self.prompt_tokens)

        # Global chunk = world_size ranks * per-rank block size.  Ring prefill
        # divides each chunk among ranks, so a chunk smaller than world_size
        # tokens cannot produce non-empty rank blocks.
        self._max_chunk_size = self._world_size * ring_prefill_block_size(self.model)

        # Build chunk range boundaries for all tokens.  If the tail chunk is
        # too small to split into non-empty rank blocks, merge it into the
        # previous chunk (matching the eager ring-prefill path logic).
        chunk_ranges = list(range(0, self._total_tokens, self._max_chunk_size))
        if (
            len(chunk_ranges) > 1
            and self._total_tokens - chunk_ranges[-1] < self._world_size
        ):
            chunk_ranges.pop()
        self._chunk_ranges = chunk_ranges

        self._t_start = time.perf_counter()

        stall_timeout_seconds = float(
            os.environ.get("EXO_PREFILL_STALL_TIMEOUT", "300")
        )
        self._watchdog = StallWatchdog(stall_timeout_seconds, "Chunked ring prefill")

        logger.info(
            f"ChunkedRingPrefill: {self._total_tokens} tokens in "
            f"{len(self._chunk_ranges)} chunks of {self._max_chunk_size} "
            f"(block_size={ring_prefill_block_size(self.model)}, rank={self._rank})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def all_done(self) -> bool:
        return self._all_done

    @property
    def chunks_remaining(self) -> int:
        return 0 if self._all_done else len(self._chunk_ranges) - self._chunk_index

    def advance_chunk(self) -> None:
        """Process exactly one ring prefill chunk.

        Each rank runs its local slice of the current chunk through the model;
        RingAttentionLayer handles cross-rank KV rotation internally.  The
        prefill flag is held only for the duration of this chunk, so decode
        steps can run between chunks without ring behaviour.
        """
        if self._all_done:
            raise RuntimeError("advance_chunk called after all chunks complete")

        chunk_start = self._chunk_ranges[self._chunk_index]
        chunk_end = min(chunk_start + self._max_chunk_size, self._total_tokens)
        if self._chunk_index == len(self._chunk_ranges) - 1:
            chunk_end = self._total_tokens
        chunk_size = chunk_end - chunk_start

        # Per-rank token range within this chunk (mirrors the eager path).
        rank_start = chunk_start + (chunk_size * self._rank) // self._world_size
        rank_end = chunk_start + (chunk_size * (self._rank + 1)) // self._world_size

        set_ring_prefill(self.model, is_prefill=True)
        try:
            with mx.stream(generation_stream):
                self.model(self.prompt_tokens[rank_start:rank_end][None], cache=self.cache)
                mx.eval([c.state for c in self.cache])  # type: ignore
        finally:
            set_ring_prefill(self.model, is_prefill=False)

        if self._watchdog is not None:
            self._watchdog.kick()

        if self.distributed_prompt_progress_callback is not None:
            self.distributed_prompt_progress_callback()

        if self.on_prefill_progress is not None:
            self.on_prefill_progress(chunk_end, self._total_tokens)

        logger.debug(
            f"ChunkedRingPrefill: chunk {self._chunk_index + 1}/{len(self._chunk_ranges)} "
            f"done ({chunk_start}:{chunk_end} tokens, rank {self._rank})"
        )

        self._chunk_index += 1
        if self._chunk_index >= len(self._chunk_ranges):
            self._all_done = True
            self._done_at = time.perf_counter()

    def finish(
        self,
        prefix_cache_save_fn: Callable[..., None] | None = None,
    ) -> tuple[float, int]:
        """Transition from ring prefill to decode mode.

        Args:
            prefix_cache_save_fn: Optional callable to save the prefix cache
                after prefill completes.

        Returns:
            (tokens_per_sec, num_tokens)
        """
        assert self._all_done, "finish called before all chunks complete"

        if self._watchdog is not None:
            self._watchdog.close()
            self._watchdog = None

        elapsed = self._done_at - self._t_start if self._done_at > self._t_start else 0.001
        tps = self._total_tokens / elapsed

        if prefix_cache_save_fn is not None:
            try:
                prefix_cache_save_fn(
                    self.prompt_tokens,
                    list(self.cache),
                    None,
                    self.prefix_hit_length,
                    None,
                    1000,
                    [],
                    prefill_tps=tps,
                )
            except Exception:
                logger.warning(
                    "Failed to save prefix cache after chunked ring prefill",
                    exc_info=True,
                )

        logger.info(
            f"ChunkedRingPrefill complete: {self._total_tokens} tokens in "
            f"{elapsed:.2f}s ({tps:.1f} tok/s), "
            f"{len(self._chunk_ranges)} chunks processed"
        )

        return tps, self._total_tokens

    def __del__(self) -> None:
        if self._watchdog is not None:
            self._watchdog.close()


class PrefillCancelled(Exception):
    """Raised when distributed prefill is cancelled via callback."""
