"""MLX generation core.

Prefill, warmup, sampling (``mlx_generate()``), token banning, constrained processing, and top-logprob extraction."""

import contextlib
import functools
import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, cast, get_args

import mlx.core as mx
from mlx_lm.generate import (
    maybe_quantize_kv_cache,
    stream_generate,
)
from mlx_lm.models.cache import (
    BatchRotatingKVCache,
    RotatingKVCache,
)
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.api.types import (
    CompletionTokensDetails,
    FinishReason,
    GenerationStats,
    PromptTokensDetails,
    TopLogprobItem,
    Usage,
)
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory, get_system_memory_total
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
)
from exo.utils.stall_watchdog import StallWatchdog
from exo.worker.engines.mlx.auto_parallel import (
    PipelineFirstLayer,
    PipelineLastLayer,
    clear_prefill_sends,
    flush_prefill_sends,
    set_pipeline_prefill,
    set_pipeline_queue_sends,
)
from exo.worker.engines.mlx.cache import (
    MEMORY_THRESHOLD,
    CacheSnapshot,
    KVPrefixCache,
    copy_snapshot_entry,
    encode_prompt,
    get_memory_used_percentage,
    has_non_kv_caches,
    is_non_trimmable_cache_entry,
    make_kv_cache,
    measure_kv_cache_bytes_per_token,
    snapshot_ssm_states,
)
from exo.worker.engines.mlx.constants import (
    DEFAULT_TOP_LOGPROBS,
    KEEP_KV_SIZE,
    KV_CACHE_BITS,
    KV_CACHE_GROUP_SIZE,
    MAX_KV_SIZE,
    MAX_TOKENS,
)
from exo.worker.engines.mlx.generator.constrained_decoding import (
    ConstrainedDecodingProcessor,
)
from exo.worker.engines.mlx.generator.remote_prefill import remote_prefill
from exo.worker.engines.mlx.generator.stop_sequences import scan_stop_sequences
from exo.worker.engines.mlx.ring_attention import (
    ring_prefill_block_size,
    set_ring_prefill,
    uses_ring_sequence_parallel_prefill,
    validate_ring_cache,
)
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    fix_unmatched_think_end_tokens,
    mx_barrier,
    mx_ranks_agree_on_value,
    system_prompt_token_count,
)
from exo.worker.engines.mlx.vision import (
    MediaRegion,
    VisionProcessor,
    VisionResult,
    get_inner_model,
    prepare_vision,
)
from exo.worker.runner.bootstrap import logger

REMOTE_PREFILL_MIN_TOKENS = 1000

generation_stream = mx.new_stream(mx.default_device())


def effective_kv_bits(cache: KVCacheType, kv_bits: int | None) -> int | None:
    """Return ``kv_bits`` only when the cache can actually be quantized.

    mlx_lm's ``maybe_quantize_kv_cache`` calls ``c.to_quantized()`` on every
    cache entry that has the method. ``RotatingKVCache`` /
    ``BatchRotatingKVCache`` expose ``to_quantized`` but raise
    ``NotImplementedError("... Quantization NYI")`` — so passing
    ``kv_bits=EXO_KV_CACHE_BITS`` with the default rotating cache (T9 #1860,
    ``MAX_KV_SIZE=16384``) crashes every generate/prefill call. Return ``None``
    for rotating caches so mlx_lm skips quantization entirely. Non-rotating
    caches (``KVCache``, model ``make_cache`` results) keep quantization.
    """
    if kv_bits is None:
        return None
    for c in cache:
        if isinstance(c, (RotatingKVCache, BatchRotatingKVCache)):
            logger.warning(
                "EXO_KV_CACHE_BITS=%s ignored: rotating KV cache does not support "
                "quantization (mlx_lm RotatingKVCache.to_quantized NYI).",
                kv_bits,
            )
            return None
    return kv_bits


def compute_ring_prefill_chunks(
    num_tokens: int,
    world_size: int,
    max_chunk_size: int,
) -> list[tuple[int, int]]:
    """Compute ring prefill chunk boundaries.

    Each chunk is ``max_chunk_size`` tokens (except possibly the last). The
    boundaries are global token indices: ``(chunk_start, chunk_end)`` where
    ``chunk_end`` is exclusive. A trailing partial chunk shorter than
    ``world_size`` tokens is merged into the previous chunk so every rank
    keeps a non-empty block.

    Mirrors the loop previously inlined in ``prefill()``; pure and testable
    without a distributed group.
    """
    chunk_starts = list(range(0, num_tokens, max_chunk_size))
    if len(chunk_starts) > 1 and num_tokens - chunk_starts[-1] < world_size:
        chunk_starts.pop()

    chunks: list[tuple[int, int]] = []
    for chunk_start in chunk_starts:
        chunk_end = min(chunk_start + max_chunk_size, num_tokens)
        if chunk_start == chunk_starts[-1]:
            chunk_end = num_tokens
        chunks.append((chunk_start, chunk_end))
    return chunks


def compute_ring_prefill_rank_bounds(
    chunk_start: int,
    chunk_end: int,
    rank: int,
    world_size: int,
) -> tuple[int, int]:
    """Compute the per-rank token slice for one ring prefill chunk."""
    chunk_size = chunk_end - chunk_start
    start = chunk_start + (chunk_size * rank) // world_size
    end = chunk_start + (chunk_size * (rank + 1)) // world_size
    return (start, end)


@dataclass
class RingPrefillState:
    """Mutable state for interleaving ring prefill with decode.

    Holds the pending prompt tokens, the KV cache being filled, the list of
    chunk ranges, and a cursor into it. ``step()`` processes exactly one
    chunk and returns False when all chunks are done.
    """

    prompt_tokens: mx.array
    cache: KVCacheType
    group: mx.distributed.Group
    model: Model
    chunk_ranges: list[tuple[int, int]]
    rank: int
    world_size: int
    index: int = 0
    stall_watchdog: StallWatchdog | None = None

    def step(self) -> bool:
        """Process one ring prefill chunk.

        Returns True if a chunk was processed, False when exhausted.
        """
        if self.index >= len(self.chunk_ranges):
            return False
        chunk_start, chunk_end = self.chunk_ranges[self.index]
        start, end = compute_ring_prefill_rank_bounds(
            chunk_start, chunk_end, self.rank, self.world_size
        )
        with mx.stream(generation_stream):
            self.model(self.prompt_tokens[start:end][None], cache=self.cache)
            mx.eval([c.state for c in self.cache])  # type: ignore
        if self.stall_watchdog is not None:
            self.stall_watchdog.kick()
        self.index += 1
        return True


def make_ring_prefill_state(
    model: Model,
    prompt_tokens: mx.array,
    cache: KVCacheType,
    group: mx.distributed.Group,
    stall_timeout_seconds: float | None = None,
) -> RingPrefillState:
    """Build chunked ring prefill state for interleaved execution.

    Mirrors the chunk computation in ``prefill()``; used by
    ``ExoBatchGenerator`` so long ring prompts prefill one chunk per decode
    step instead of blocking ``submit()``.
    """
    num_tokens = len(prompt_tokens)
    rank = group.rank()
    world_size = group.size()
    max_chunk_size = world_size * ring_prefill_block_size(model)
    chunk_ranges = compute_ring_prefill_chunks(num_tokens, world_size, max_chunk_size)
    if stall_timeout_seconds is None:
        stall_timeout_seconds = float(os.environ.get("EXO_PREFILL_STALL_TIMEOUT", "300"))
    return RingPrefillState(
        model=model,
        prompt_tokens=prompt_tokens,
        cache=cache,
        group=group,
        chunk_ranges=chunk_ranges,
        rank=rank,
        world_size=world_size,
        stall_watchdog=StallWatchdog(
            stall_timeout_seconds, f"Ring prefill ({num_tokens} tokens)"
        ),
    )


@contextlib.contextmanager
def patch_embed_tokens(
    model: Model,
    embeddings: mx.array,
    start_offset: int = 0,
    token_count: int = 0,
    image_token_id: int | None = None,
) -> Generator[None]:
    """Temporarily splice precomputed embeddings into the model's embed layer.

    Within the context, token positions in ``[start_offset, start_offset +
    token_count)`` are replaced by the given ``embeddings`` (used for image /
    media injection); the original embed function is restored on exit.
    """
    inner = get_inner_model(model)  # type: ignore
    original_embed = inner.embed_tokens  # type: ignore
    end_offset = start_offset + token_count
    offset = [start_offset]

    def _inject(input_ids: mx.array) -> mx.array:
        chunk_start = offset[0]
        chunk_len = input_ids.shape[-1]
        chunk_end = chunk_start + chunk_len
        offset[0] = chunk_end

        # The injection window is [start_offset, end_offset).
        if chunk_end <= start_offset or chunk_start >= end_offset:
            return original_embed(input_ids)  # type: ignore

        # Mixed chunk: splice the pre-computed embeddings for the overlap
        # into `original_embed(input_ids)` for any text-only fringes.
        overlap_start = max(chunk_start, start_offset)
        overlap_end = min(chunk_end, end_offset)
        dst_start = overlap_start - chunk_start
        dst_end = overlap_end - chunk_start
        text_embeds: mx.array = original_embed(input_ids)  # type: ignore
        return mx.concatenate(
            [
                text_embeds[:, :dst_start, :],
                embeddings[:, overlap_start:overlap_end, :],
                text_embeds[:, dst_end:, :],
            ],
            axis=1,
        )

    for attr in dir(original_embed):  # type: ignore
        if not attr.startswith("_") and not hasattr(_inject, attr):
            with contextlib.suppress(AttributeError, TypeError):
                setattr(_inject, attr, getattr(original_embed, attr))  # type: ignore

    inner.embed_tokens = _inject

    # Gemma 4 (e2b/e4b) has a second, independent embedding table that produces
    # per-layer conditioning signals via self.embed_tokens_per_layer(input_ids).
    # The injected vision embeddings live in the main residual stream only, so
    # if image_token_id positions are passed through as-is the per-layer table
    # produces garbage signals at those positions (the `<image>` token was never
    # trained to have meaningful per-layer inputs).
    original_per_layer = getattr(inner, "embed_tokens_per_layer", None)  # type: ignore
    if original_per_layer is not None and image_token_id is not None:

        def _clean_per_layer(input_ids: mx.array) -> mx.array:
            clean_ids = mx.where(
                input_ids == image_token_id, mx.zeros_like(input_ids), input_ids
            )
            return original_per_layer(clean_ids)  # type: ignore

        inner.embed_tokens_per_layer = _clean_per_layer

    try:
        yield
    finally:
        inner.embed_tokens = original_embed
        if original_per_layer is not None and image_token_id is not None:
            inner.embed_tokens_per_layer = original_per_layer


class PrefillCancelled(BaseException):
    """Raised when prefill is cancelled via the progress callback."""


def _has_pipeline_communication_layer(model: Model):
    for layer in model.layers:
        if isinstance(layer, (PipelineFirstLayer, PipelineLastLayer)):
            return True
    return False


def pipeline_parallel_prefill(
    model: Model,
    prompt: mx.array,
    prompt_cache: KVCacheType,
    prefill_step_size: int,
    kv_group_size: int | None,
    kv_bits: int | None,
    prompt_progress_callback: Callable[[int, int], None],
    distributed_prompt_progress_callback: Callable[[], None] | None,
    group: mx.distributed.Group,
) -> None:
    """Prefill the KV cache for pipeline parallel with overlapping stages.

    Each rank processes the full prompt through its real cache, offset by leading
    and trailing dummy iterations.

    Total iterations per rank = N_real_chunks + world_size - 1:
      - rank r leading dummies  (skip_pipeline_io, throwaway cache)
      - N_real_chunks real      (pipeline IO active, real cache)
      - (world_size-1-r) trailing dummies (skip_pipeline_io, throwaway cache)

    e.g.
    Timeline (2 ranks, 3 chunks of 10240 tokens @ step=4096):
        iter 0: R0 real[0:4096]     R1 dummy
        iter 1: R0 real[4096:8192]  R1 real[0:4096]
        iter 2: R0 real[8192:10240] R1 real[4096:8192]
        iter 3: R0 dummy            R1 real[8192:10240]

    This function is designed to match mlx_lm's stream_generate exactly in terms of
    side effects (given the same prefill step size)
    """
    prefill_step_size = prefill_step_size // min(4, group.size())

    quantize_cache_fn: Callable[..., None] = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=0,
        kv_group_size=kv_group_size,
        kv_bits=effective_kv_bits(prompt_cache, kv_bits),
    )

    _prompt_cache: KVCacheType = prompt_cache
    rank = group.rank()
    world_size = group.size()

    # Build list of real prompt chunk sizes
    total = len(prompt)
    real_chunk_sizes: list[int] = []
    remaining = total - 1
    while remaining:
        n = min(prefill_step_size, remaining)
        real_chunk_sizes.append(n)
        remaining -= n
    n_real = len(real_chunk_sizes)

    # Each rank does: [rank leading dummies] [N real chunks] [world_size-1-rank trailing dummies]
    n_leading = rank
    n_trailing = world_size - 1 - rank
    n_total = n_leading + n_real + n_trailing

    t_start = time.perf_counter()
    processed = 0
    logger.info(
        f"[R{rank}] Pipeline prefill: {n_real} real + {n_leading} leading + {n_trailing} trailing = {n_total} iterations"
    )
    clear_prefill_sends()

    # Initial callback matching generate_step
    prompt_progress_callback(0, total)

    try:
        with mx.stream(generation_stream):
            for _ in range(n_leading):
                if distributed_prompt_progress_callback is not None:
                    distributed_prompt_progress_callback()

            for i in range(n_real):
                chunk_size = real_chunk_sizes[i]
                model(
                    prompt[processed : processed + chunk_size][None],
                    cache=_prompt_cache,
                )
                quantize_cache_fn(_prompt_cache)
                processed += chunk_size

                if distributed_prompt_progress_callback is not None:
                    distributed_prompt_progress_callback()

                flush_prefill_sends()

                prompt_progress_callback(processed, total)

            for _ in range(n_trailing):
                if distributed_prompt_progress_callback is not None:
                    distributed_prompt_progress_callback()

    finally:
        clear_prefill_sends()

    # Post-loop: process remaining 1 token + add +1 entry to match stream_generate.
    for _ in range(2):
        with mx.stream(generation_stream):
            model(prompt[-1:][None], cache=_prompt_cache)
            quantize_cache_fn(_prompt_cache)
        flush_prefill_sends()

    assert _prompt_cache is not None
    with mx.stream(generation_stream):
        mx.eval([c.state for c in _prompt_cache])  # type: ignore

    # Final callback matching generate_step
    prompt_progress_callback(total, total)

    logger.info(
        f"[R{rank}] Prefill: {n_real} real + {n_leading}+{n_trailing} dummy iterations, "
        f"Processed {processed} tokens in {(time.perf_counter() - t_start) * 1000:.1f}ms"
    )


def prefill(
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
    prompt_tokens: mx.array,
    cache: KVCacheType,
    group: mx.distributed.Group | None,
    on_prefill_progress: Callable[[int, int], None] | None,
    distributed_prompt_progress_callback: Callable[[], None] | None,
) -> tuple[float, int, list[CacheSnapshot]]:
    """Prefill the KV cache with prompt tokens.

    This runs the model over the prompt tokens to populate the cache,
    then trims off the extra generated token.

    Returns:
        (tokens_per_sec, num_tokens, snapshots)
    """
    num_tokens = len(prompt_tokens)
    if num_tokens == 0:
        return 0.0, 0, []

    logger.debug(f"Prefilling {num_tokens} tokens...")
    start_time = time.perf_counter()
    has_ssm = has_non_kv_caches(cache)
    snapshots: list[CacheSnapshot] = []

    # TODO(evan): kill the callbacks/runner refactor
    def progress_callback(processed: int, total: int) -> None:
        """Log prefill progress at debug level + snapshot SSM states."""
        elapsed = time.perf_counter() - start_time
        tok_per_sec = processed / elapsed if elapsed > 0 else 0
        logger.debug(
            f"Prefill progress: {processed}/{total} tokens ({tok_per_sec:.1f} tok/s)"
        )
        if has_ssm:
            snapshots.append(snapshot_ssm_states(cache))

        if on_prefill_progress is not None:
            on_prefill_progress(processed, total)

    def combined_progress_callback(processed: int, total: int) -> None:
        """Distributed-prompt signal plus local progress callback."""
        if distributed_prompt_progress_callback is not None:
            distributed_prompt_progress_callback()
        progress_callback(processed, total)

    is_ring_prefill = uses_ring_sequence_parallel_prefill(model, num_tokens, group)
    if is_ring_prefill:
        validate_ring_cache(cache)
    set_pipeline_prefill(model, is_prefill=True)
    set_ring_prefill(model, is_prefill=is_ring_prefill)

    is_pipeline = _has_pipeline_communication_layer(model)

    prefill_step_size = int(os.getenv("EXO_PREFILL_STEP_SIZE", "4096"))

    try:
        mx_barrier(group)
        logger.info("Starting prefill")

        if is_ring_prefill:
            assert group is not None
            ring_state = make_ring_prefill_state(
                model, prompt_tokens, cache, group
            )
            combined_progress_callback(0, num_tokens)
            with ring_state.stall_watchdog or contextlib.nullcontext():
                while ring_state.step():
                    combined_progress_callback(
                        ring_state.chunk_ranges[ring_state.index - 1][1], num_tokens
                    )
        elif is_pipeline and num_tokens >= prefill_step_size:
            set_pipeline_queue_sends(model, queue_sends=True)
            assert group is not None, "Pipeline prefill requires a distributed group"
            pipeline_parallel_prefill(
                model=model,
                prompt=prompt_tokens,
                prompt_cache=cache,
                prefill_step_size=prefill_step_size,
                kv_group_size=KV_CACHE_GROUP_SIZE,
                kv_bits=KV_CACHE_BITS,
                prompt_progress_callback=progress_callback,
                distributed_prompt_progress_callback=distributed_prompt_progress_callback,
                group=group,
            )
        else:
            # Use max_tokens=1 because max_tokens=0 does not work.
            # We just throw away the generated token - we only care about filling the cache
            for _ in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_tokens,
                max_tokens=1,
                sampler=sampler,
                prompt_cache=cache,
                prefill_step_size=prefill_step_size,
                kv_group_size=KV_CACHE_GROUP_SIZE,
                kv_bits=effective_kv_bits(cache, KV_CACHE_BITS),
                prompt_progress_callback=combined_progress_callback,
            ):
                break  # Stop after first iteration - cache is now filled
    finally:
        set_pipeline_queue_sends(model, queue_sends=False)
        set_pipeline_prefill(model, is_prefill=False)
        set_ring_prefill(model, is_prefill=False)

    # Barrier after prefill to prevent ranks from entering generation (all_gather)
    # while other ranks are still in prefill (skip all_gather).
    if is_pipeline:
        mx_barrier(group)

    if not is_ring_prefill:
        # stream_generate added 1 extra generated token to the cache, so we should trim it.
        # Because of needing to roll back arrays cache, we will generate on 2 tokens so trim 1 more.
        pre_gen = snapshots[-2] if has_ssm else None
        for i, c in enumerate(cache):
            non_trimmable = is_non_trimmable_cache_entry(c)
            if has_ssm and non_trimmable:
                assert pre_gen is not None
                restored = copy_snapshot_entry(pre_gen.states[i])
                if restored is not None:
                    cache[i] = restored  # type: ignore
            else:
                assert not non_trimmable
                c.trim(2)

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0.0
    logger.debug(
        f"Prefill complete: {num_tokens} tokens in {elapsed:.2f}s "
        f"({tokens_per_sec:.1f} tok/s)"
    )
    # Exclude the last snapshot
    return tokens_per_sec, num_tokens, snapshots[:-1] if snapshots else []


def warmup_inference(
    model: Model,
    tokenizer: TokenizerWrapper,
    group: mx.distributed.Group | None,
    model_id: ModelId,
) -> tuple[int, int]:
    """Run a short warmup generation so the engine is primed before use.

    Returns a tuple of ``(check_for_cancel_every, bytes_per_token)`` where
    ``bytes_per_token`` is the measured KV-cache memory consumption per token
    used by the OOM-prevention budget check (#1626).
    """
    logger.info(f"warming up inference for instance: {model_id}")

    content = InputMessageContent(
        "Prompt to warm up the inference engine. Repeat this."
    )

    warmup_task_params = TextGenerationTaskParams(
        model=model_id,
        input=[InputMessage(role="user", content=content)],
        max_output_tokens=50,
        temperature=0.0,
    )

    warmup_prompt = apply_chat_template(
        tokenizer=tokenizer,
        task_params=warmup_task_params,
    )

    tokens_generated = 0

    mx_barrier(group)

    logger.info("Generating warmup tokens")

    t = time.monotonic()

    for _r in mlx_generate(
        model=model,
        tokenizer=tokenizer,
        task=warmup_task_params,
        prompt=warmup_prompt,
        kv_prefix_cache=None,
        group=group,
    ):
        tokens_generated += 1

    check_for_cancel_every = min(
        math.ceil(tokens_generated / min(time.monotonic() - t, 0.001)), 100
    )

    mx_barrier(group)

    logger.info(f"warmed up by generating {tokens_generated} tokens")
    if group is not None:
        check_for_cancel_every = int(
            mx.max(
                mx.distributed.all_gather(
                    mx.array([check_for_cancel_every]),
                    group=group,
                )
            ).item()
        )

    logger.info(
        f"runner checking for cancellation every {check_for_cancel_every} tokens"
    )

    # Measure KV-cache bytes per token from the warmup run (OOM prevention).
    # We count the prompt + generated tokens; the warmup KV cache is short-lived
    # and freed on model teardown, but the per-token figure is stable for a given
    # model/layer config and is what the budget check multiplies by sequence
    # length to predict growth.
    try:
        if hasattr(model, "layers"):
            warmup_cache = [layer.cache for layer in model.layers if hasattr(layer, "cache")]
            if warmup_cache:
                bpt = measure_kv_cache_bytes_per_token(warmup_cache)  # type: ignore[arg-type]
            else:
                bpt = 0
        else:
            bpt = 0
    except Exception:
        bpt = 0  # measurement is best-effort; 0 disables the budget check
    if bpt > 0:
        logger.info(f"measured KV cache memory: {bpt} bytes/token")

    return check_for_cancel_every, bpt


def _check_memory_budget(
    bytes_per_token: int,
    total_sequence_tokens: int,
    kv_prefix_cache: KVPrefixCache | None,
) -> str | None:
    """Check if enough memory is available for the estimated KV cache.

    Uses the same memory pressure system as prefix cache eviction.
    If memory would exceed the threshold, tries evicting prefix caches first.

    Returns None if OK, or an error message string if OOM is predicted.
    """
    if bytes_per_token == 0:
        return None

    total_ram = get_system_memory_total().in_bytes
    estimated_cache_bytes = bytes_per_token * total_sequence_tokens
    current_pressure = (
        kv_prefix_cache.get_memory_used_percentage()
        if kv_prefix_cache is not None
        else get_memory_used_percentage()
    )
    projected_pressure = current_pressure + (estimated_cache_bytes / total_ram)

    logger.info(
        f"Memory check: {total_sequence_tokens} tokens × {bytes_per_token} B/tok "
        f"= {estimated_cache_bytes / (1024**2):.1f} MB, "
        f"pressure {current_pressure:.1%} → projected {projected_pressure:.1%} "
        f"(threshold {MEMORY_THRESHOLD:.1%})"
    )

    if projected_pressure <= MEMORY_THRESHOLD:
        return None

    # Try evicting all prefix caches
    if kv_prefix_cache is not None:
        evicted = kv_prefix_cache.force_evict_all()
        if evicted > 0:
            mx.clear_cache()
            current_pressure = kv_prefix_cache.get_memory_used_percentage()
            projected_pressure = current_pressure + (estimated_cache_bytes / total_ram)
            if projected_pressure <= MEMORY_THRESHOLD:
                return None

    return (
        f"Not enough memory: projected KV cache pressure would reach "
        f"{projected_pressure:.1%} (threshold {MEMORY_THRESHOLD:.1%}). "
        f"Estimated need: {estimated_cache_bytes / (1024**2):.1f} MB for "
        f"{total_sequence_tokens} tokens at {bytes_per_token} B/token."
    )


def ban_token_ids(token_ids: list[int]) -> Callable[[mx.array, mx.array], mx.array]:
    """Build a logits processor that forbids the given token ids."""
    token_ids = [int(t) for t in token_ids]

    def proc(_history: mx.array, logits: mx.array) -> mx.array:
        """Set forbidden token logits to a very negative value."""
        for tid in token_ids:
            logits[..., tid] = -1e9
        return logits

    return proc


def eos_ids_from_tokenizer(tokenizer: TokenizerWrapper) -> list[int]:
    """Return the tokenizer's EOS token ids (empty list if none declared)."""
    eos: list[int] | None = getattr(tokenizer, "eos_token_ids", None)
    if eos is None:
        return []
    return eos


def make_constrained_processor(
    task: TextGenerationTaskParams,
    tokenizer: TokenizerWrapper,
) -> Callable[[mx.array, mx.array], mx.array] | None:
    """Build a JSON-schema constrained logits processor, or None if not requested.

    T28: when ``task.response_format`` is set, sampling is masked so the
    output must match the JSON Schema. Unsupported/invalid schemas fail open
    with a warning (never crash mid-generation); the API validates schemas at
    request time for a clean 400.
    """
    if task.response_format is None:
        return None
    schema: object = task.response_format
    if isinstance(schema, str):
        try:
            parsed = cast("object", json.loads(schema))
        except ValueError:
            logger.warning(
                "Constrained decoding: invalid JSON schema string — ignoring"
            )
            return None
        if not isinstance(parsed, dict):
            logger.warning(
                "Constrained decoding: response_format must be a JSON Schema object — ignoring"
            )
            return None
        schema = cast("dict[str, object]", parsed)
    if not isinstance(schema, dict):  # type: ignore[reportUnnecessaryIsInstance]  # runtime guard (test bypasses pydantic)
        logger.warning(
            "Constrained decoding: response_format must be a JSON Schema object — ignoring"
        )
        return None
    schema_dict = cast("dict[str, Any]", schema)
    # OpenAI-style wrapper: {"type": "json_object", "schema": {...}}.
    # The actual JSON Schema lives under the "schema" key; without this
    # unwrap the wrapper dict would be compiled as a "string"-typed value
    # (unknown type -> _string_fsm) and the output would start with '"'.
    inner_schema: object = schema_dict.get("schema")
    if isinstance(inner_schema, dict):
        schema_dict = cast("dict[str, Any]", inner_schema)
    elif isinstance(schema_dict.get("json_schema"), dict):
        # OpenAI structured outputs: {"type": "json_schema", "json_schema": {"schema": {...}}}
        inner = cast("dict[str, Any]", schema_dict["json_schema"])
        inner_schema = inner.get("schema")
        if isinstance(inner_schema, dict):
            schema_dict = cast("dict[str, Any]", inner_schema)
    elif schema_dict.get("type") == "json_object":
        # OpenAI json_object mode: "any valid JSON object" — no schema is
        # attached, so compile as a bare object instead of treating the
        # wrapper as an unknown type (which would fall back to a string FSM
        # and force the output to start with a quote).
        schema_dict = {"type": "object"}
    try:
        return ConstrainedDecodingProcessor(tokenizer, schema_dict)
    except ValueError:
        logger.warning(
            "Constrained decoding: unsupported schema keywords — ignoring (request "
            "time validation should have rejected this earlier)"
        )
        return None


def extract_top_logprobs(
    logprobs: mx.array,
    tokenizer: TokenizerWrapper,
    top_logprobs: int,
    selected_token: int,
    precomputed_indices: list[int] | None = None,
    precomputed_values: list[float] | None = None,
    precomputed_selected: float | None = None,
) -> tuple[float, list[TopLogprobItem]]:
    """Extract the selected token's logprob plus the top-k alternatives.

    ``precomputed_*`` args skip re-argpartition when the caller already
    computed the top-k indices/values for this position. NaN logprobs are
    skipped. Returns ``(selected_logprob, top_logprob_items)``.
    """
    if (
        precomputed_indices is not None
        and precomputed_values is not None
        and precomputed_selected is not None
    ):
        top_indices_list: list[int] = precomputed_indices[:top_logprobs]
        top_values_list: list[float] = precomputed_values[:top_logprobs]
        selected_logprob = precomputed_selected
    else:
        selected_logprob_arr = logprobs[selected_token]
        top_logprobs = min(top_logprobs, logprobs.shape[0] - 1)
        top_indices = mx.argpartition(-logprobs, top_logprobs)[:top_logprobs]
        top_values = logprobs[top_indices]
        sort_order = mx.argsort(-top_values)
        top_indices = top_indices[sort_order]
        top_values = top_values[sort_order]
        mx.eval(selected_logprob_arr, top_indices, top_values)
        selected_logprob = float(selected_logprob_arr.item())
        top_indices_list = top_indices.tolist()  # type: ignore
        top_values_list = top_values.tolist()  # type: ignore

    # Convert to list of TopLogprobItem
    top_logprob_items: list[TopLogprobItem] = []
    for token_id, token_logprob in zip(top_indices_list, top_values_list, strict=True):
        if math.isnan(token_logprob):
            continue

        # Decode token ID to string
        token_str = tokenizer.decode([token_id])
        top_logprob_items.append(
            TopLogprobItem(
                token=token_str,
                logprob=token_logprob,
                bytes=list(token_str.encode("utf-8")),
            )
        )

    return selected_logprob, top_logprob_items


def mlx_generate(
    model: Model,
    tokenizer: TokenizerWrapper,
    task: TextGenerationTaskParams,
    prompt: str,
    kv_prefix_cache: KVPrefixCache | None,
    group: mx.distributed.Group | None,
    on_prefill_progress: Callable[[int, int], None] | None = None,
    distributed_prompt_progress_callback: Callable[[], None] | None = None,
    on_generation_token: Callable[[], None] | None = None,
    vision_processor: VisionProcessor | None = None,
    bytes_per_token: int = 0,
) -> Generator[GenerationResponse]:
    """Run a single text-generation task on this model and yield tokens.

    Handles prompt encoding (with think-tag fixing and optional vision
    pre-processing), prefix-cache lookup (skipped for benchmarks or when
    ``task.use_prefix_cache`` is false), ring or pipeline prefill, and the
    decode loop, yielding a ``GenerationResponse`` per generated token.
    """
    # Ensure that generation stats only contains peak memory for this generation
    mx.reset_peak_memory()
    # TODO: Randomise task seed and set in taskparams, instead of hard coding as 42.
    seed = task.seed or 42
    mx.random.seed(seed)

    # Encode prompt once at the top and fix unmatched think tags
    all_prompt_tokens = encode_prompt(tokenizer, prompt)
    all_prompt_tokens = fix_unmatched_think_end_tokens(all_prompt_tokens, tokenizer)
    min_prefix_hit_length = max(1000, system_prompt_token_count(task, tokenizer))

    vision: VisionResult | None = None
    if vision_processor is not None:
        try:
            vision = prepare_vision(
                images=task.images,
                chat_template_messages=task.chat_template_messages,
                vision_processor=vision_processor,
                tokenizer=tokenizer,
                model=model,
                model_id=task.model,
                task_params=task,
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Vision processing failed, falling back to text-only"
            )
    if vision is not None:
        all_prompt_tokens = vision.prompt_tokens
    media_regions: list[MediaRegion] = vision.media_regions if vision else []

    # Do not use the prefix cache if we are trying to do benchmarks.
    # Skip the prefix cache when the request opts out (use_prefix_cache=False),
    # e.g. benchmarks or one-off requests that must leave no cache trace.
    if not task.use_prefix_cache:
        kv_prefix_cache = None

    # Use prefix cache if available, otherwise create fresh cache
    prefix_hit_length = 0
    matched_index: int | None = None
    is_exact_hit = False
    if kv_prefix_cache is None:
        caches = make_kv_cache(
            model=model, max_kv_size=MAX_KV_SIZE, keep=KEEP_KV_SIZE or 0
        )
        prompt_tokens = all_prompt_tokens
    else:
        caches, prompt_tokens, matched_index, is_exact_hit = (
            kv_prefix_cache.get_kv_cache(
                model, all_prompt_tokens, media_regions=media_regions
            )
        )
        prefix_hit_length = len(all_prompt_tokens) - len(prompt_tokens)
        if not mx_ranks_agree_on_value(prefix_hit_length, group):
            # Divergent restore positions would make ranks prefill different
            # token counts and deadlock the pipeline.
            logger.warning(
                "KV prefix cache hit lengths diverge across pipeline ranks; "
                "discarding the hit to keep prefill in lockstep"
            )
            caches = make_kv_cache(
                model=model, max_kv_size=MAX_KV_SIZE, keep=KEEP_KV_SIZE or 0
            )
            prompt_tokens = all_prompt_tokens
            prefix_hit_length = 0
            matched_index = None
            is_exact_hit = False
        elif prefix_hit_length > 0:
            logger.info(
                f"KV cache hit: {prefix_hit_length}/{len(all_prompt_tokens)} tokens cached ({100 * prefix_hit_length / len(all_prompt_tokens):.1f}%)"
            )

    logits_processors: list[Callable[[mx.array, mx.array], mx.array]] = (
        make_logits_processors(
            logit_bias=task.logit_bias,
            repetition_penalty=task.repetition_penalty,
            repetition_context_size=task.repetition_context_size
            if task.repetition_context_size is not None
            else 20,
            presence_penalty=task.presence_penalty,
            frequency_penalty=task.frequency_penalty,
        )
    )
    if task.bench:
        # Only sample length eos tokens
        eos_ids = eos_ids_from_tokenizer(tokenizer)
        logits_processors = [ban_token_ids(eos_ids)] + logits_processors

    # T28: JSON-schema constrained decoding (fail-open on unsupported schemas).
    constrained = make_constrained_processor(task, tokenizer)
    if constrained is not None:
        logits_processors = [constrained] + logits_processors

    sampler = make_sampler(
        temp=task.temperature if task.temperature is not None else 0.7,
        top_p=task.top_p if task.top_p is not None else 1.0,
        min_p=task.min_p if task.min_p is not None else 0.05,
        top_k=task.top_k if task.top_k is not None else 0,
    )

    # Normalize stop sequences to a list
    stop_sequences: list[str] = (
        ([task.stop] if isinstance(task.stop, str) else task.stop)
        if task.stop is not None
        else []
    )

    maybe_vision_ctx = (
        patch_embed_tokens(
            model,
            vision.embeddings,
            prefix_hit_length,
            len(prompt_tokens) - 1,
            image_token_id=vision.image_token_id,
        )
        if vision is not None
        else contextlib.nullcontext()
    )
    use_remote = (
        len(prompt_tokens) > REMOTE_PREFILL_MIN_TOKENS
        and task.prefill_endpoint is not None
    )
    remote_prefilled = False
    prefill_tps = 0.0
    prefill_tokens = 0
    ssm_snapshots_list: list[CacheSnapshot] = []
    with maybe_vision_ctx:
        if use_remote and task.prefill_endpoint is not None:
            try:
                prefill_tps, prefill_tokens, ssm_snapshots_list = remote_prefill(
                    prompt_tokens[:-1],
                    caches,
                    on_prefill_progress,
                    endpoint=task.prefill_endpoint,
                    request_id=str(uuid.uuid4()),
                    model_id=str(task.model),
                    start_pos=prefix_hit_length,
                )
                remote_prefilled = True
            except Exception:
                logger.opt(exception=True).warning(
                    "Remote prefill failed, falling back to local prefill"
                )
        if not remote_prefilled:
            prefill_tps, prefill_tokens, ssm_snapshots_list = prefill(
                model,
                tokenizer,
                sampler,
                prompt_tokens[:-1],
                caches,
                group,
                on_prefill_progress,
                distributed_prompt_progress_callback,
            )
    cache_snapshots: list[CacheSnapshot] | None = ssm_snapshots_list or None

    if kv_prefix_cache is not None and matched_index is not None and is_exact_hit:
        prefill_tps = kv_prefix_cache.prefill_tps[matched_index]

    if kv_prefix_cache is not None:
        if kv_prefix_cache.should_update_entry(
            matched_index, prefix_hit_length, min_prefix_hit_length
        ):
            assert matched_index is not None
            kv_prefix_cache.update_kv_cache(
                matched_index,
                all_prompt_tokens,
                caches,
                cache_snapshots,
                restore_pos=prefix_hit_length,
                media_regions=media_regions,
                prefill_tps=prefill_tps,
            )
        else:
            kv_prefix_cache.add_kv_cache(
                all_prompt_tokens,
                caches,
                cache_snapshots,
                media_regions=media_regions,
                prefill_tps=prefill_tps,
            )

    # stream_generate starts from the last token
    last_token = (
        prompt_tokens[-1:]
        if uses_ring_sequence_parallel_prefill(model, len(prompt_tokens) - 1, group)
        else prompt_tokens[-2:]
    )

    max_tokens = task.max_output_tokens or MAX_TOKENS
    # OOM prevention (#1626): predict KV-cache growth before the decode loop.
    # The prompt has already been prefilled and cached, so the remaining
    # growth is roughly bytes_per_token × (max_tokens to be generated).
    # If the projection would blow the memory budget, emit a friendly
    # error chunk instead of crashing the process with a hard OOM.
    memory_error = _check_memory_budget(
        bytes_per_token,
        len(prompt_tokens) - 1 + max_tokens,
        kv_prefix_cache,
    )
    if memory_error is not None:
        logger.warning(memory_error)
        yield GenerationResponse(
            text=memory_error,
            token=0,
            finish_reason="error",
            usage=None,
        )
        return
    # Text decoded but not yet emitted because it could be the start of a stop
    # sequence spanning multiple tokens. See scan_stop_sequences.
    pending_stop_text = ""
    generated_text_parts: list[str] = []
    generation_start_time = time.perf_counter()
    usage: Usage | None = None
    logger.info("Starting decode")
    mx_barrier(group)

    for completion_tokens, out in enumerate(
        stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=last_token,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prompt_cache=caches,
            prefill_step_size=1,
            kv_group_size=KV_CACHE_GROUP_SIZE,
            kv_bits=effective_kv_bits(caches, KV_CACHE_BITS),
        ),
        start=1,
    ):
        generated_text_parts.append(out.text)

        # Check for stop sequences, holding back any trailing partial match so a
        # multi-token stop sequence never leaks its leading bytes into output.
        model_finish_reason = cast(FinishReason | None, out.finish_reason)
        pending_stop_text += out.text
        text, matched_stop_sequence, pending_stop_text = scan_stop_sequences(
            pending_stop_text, stop_sequences
        )

        finish_reason: FinishReason | None
        if matched_stop_sequence is not None:
            finish_reason = "stop"
        elif model_finish_reason is not None:
            # Natural EOS / length limit: flush any held-back text — it is real
            # output that merely looked like the start of a stop sequence.
            text += pending_stop_text
            pending_stop_text = ""
            finish_reason = model_finish_reason
        else:
            finish_reason = None

        is_done = finish_reason is not None

        stats: GenerationStats | None = None
        if is_done:
            stats = GenerationStats(
                prompt_tps=float(prefill_tps or out.prompt_tps),
                generation_tps=float(out.generation_tps),
                prompt_tokens=int(prefill_tokens + out.prompt_tokens),
                generation_tokens=int(out.generation_tokens),
                peak_memory_usage=Memory.from_gb(out.peak_memory),
            )
            if matched_stop_sequence is None and out.finish_reason not in get_args(
                FinishReason
            ):
                logger.warning(
                    f"Model generated unexpected finish_reason: {out.finish_reason}"
                )

            total_prompt_tokens = len(all_prompt_tokens)
            usage = Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_prompt_tokens + completion_tokens,
                prompt_tokens_details=PromptTokensDetails(
                    cached_tokens=prefix_hit_length
                ),
                completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0),
            )

        # Extract logprobs from the full vocabulary logprobs array
        logprob: float | None = None
        top_logprobs: list[TopLogprobItem] | None = None
        if task.logprobs:
            with mx.stream(generation_stream):
                logprob, top_logprobs = extract_top_logprobs(
                    logprobs=out.logprobs,
                    tokenizer=tokenizer,
                    top_logprobs=task.top_logprobs or DEFAULT_TOP_LOGPROBS,
                    selected_token=out.token,
                )

        if is_done:
            # Log generation stats
            generation_elapsed = time.perf_counter() - generation_start_time
            generated_tokens = len(generated_text_parts)
            generation_tps = (
                generated_tokens / generation_elapsed if generation_elapsed > 0 else 0.0
            )
            logger.debug(
                f"Generation complete: prefill {prompt_tokens} tokens @ "
                f"{prefill_tps:.1f} tok/s, generated {generated_tokens} tokens @ "
                f"{generation_tps:.1f} tok/s"
            )
        if on_generation_token is not None:
            on_generation_token()

        yield GenerationResponse(
            text=text,
            token=out.token,
            logprob=logprob,
            top_logprobs=top_logprobs,
            finish_reason=finish_reason,
            stats=stats,
            usage=usage,
            matched_stop_sequence=matched_stop_sequence,
        )

        if is_done:
            mx_barrier(group)
            break
