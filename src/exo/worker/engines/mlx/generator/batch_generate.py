"""Batched generation over the MLX engine.

:class:`ExoBatchGenerator` handles deferred prefill, top-k sampling, and stop-sequence processing for batch requests."""

import contextlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, cast

import mlx.core as mx
from mlx_lm.generate import (
    BatchGenerator as MlxBatchGenerator,
)
from mlx_lm.generate import (
    generation_stream,
)
from mlx_lm.models.cache import RotatingKVCache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import StreamingDetokenizer, TokenizerWrapper

from exo.api.types import (
    CompletionTokensDetails,
    FinishReason,
    GenerationStats,
    PromptTokensDetails,
    TopLogprobItem,
    Usage,
)
from exo.shared.types.memory import Memory
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.mlx.auto_parallel import (
    PipelineLastLayer,
    set_pipeline_token_relay,
)
from exo.worker.engines.mlx.cache import (
    CacheSnapshot,
    KVPrefixCache,
    encode_prompt,
    make_kv_cache,
)
from exo.worker.engines.mlx.constants import (
    DEFAULT_TOP_LOGPROBS,
    KEEP_KV_SIZE,
    MAX_KV_SIZE,
    MAX_TOKENS,
)
from exo.worker.engines.mlx.generator.generate import (
    RingPrefillState,
    ban_token_ids,
    eos_ids_from_tokenizer,
    extract_top_logprobs,
    make_constrained_processor,
    make_ring_prefill_state,
    patch_embed_tokens,
    prefill,
)
from exo.worker.engines.mlx.generator.remote_prefill import remote_prefill
from exo.worker.engines.mlx.generator.stop_sequences import scan_stop_sequences
from exo.worker.engines.mlx.patches.opt_batch_gen import (
    set_needs_topk,
    take_ready_topk,
)
from exo.worker.engines.mlx.ring_attention import uses_ring_sequence_parallel_prefill
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.engines.mlx.utils_mlx import (
    fix_unmatched_think_end_tokens,
    mx_ranks_agree_on_value,
    system_prompt_token_count,
)
from exo.worker.engines.mlx.vision import (
    MediaRegion,
    VisionProcessor,
    VisionResult,
    prepare_vision,
)
from exo.worker.runner.bootstrap import logger

REMOTE_PREFILL_MIN_TOKENS = 1000


def _stop_sequences(task_params: TextGenerationTaskParams) -> list[str]:
    if task_params.stop is None:
        return []
    if isinstance(task_params.stop, str):
        return [task_params.stop]
    return task_params.stop


@dataclass
class _EngineTask:
    uid: int
    task_params: TextGenerationTaskParams
    all_prompt_tokens: mx.array
    prefix_hit_length: int
    matched_index: int | None
    detokenizer: StreamingDetokenizer
    on_generation_token: Callable[[], None] | None = None
    generated_text_parts: list[str] = field(default_factory=list)
    potential_stop_sequence_text: str = ""
    completion_tokens: int = 0
    generation_start_time: float = 0.0
    prefill_tps: float = 0.0
    prefix_cache_hit: Literal["none", "partial", "exact"] = "none"
    media_regions: list[MediaRegion] = field(default_factory=list)
    first_gen_token_time: float | None = None
    last_gen_token_time: float | None = None


@dataclass
class _PendingRingInsert:
    """Deferred mlx-lm insert for a ring-prefill task.

    Ring sequence-parallel prefill runs *outside* ``MlxBatchGenerator`` (its
    ``PromptProcessingBatch`` is not ring-aware), but must not block
    ``submit()`` the way the eager path does.  Instead we hold the task here
    and run one ``ring_state.step()`` chunk per ``ExoBatchGenerator.step()``
    call; once the cache is fully populated we finally insert the last-token
    decode seed into ``MlxBatchGenerator`` with the populated cache.
    """

    uid: int
    ring_state: RingPrefillState
    last_tokens: mx.array
    max_tokens: int
    sampler: Callable[[mx.array], mx.array]
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]]
    cache: KVCacheType
    on_prefill_progress: Callable[[int, int], None] | None
    distributed_prompt_progress_callback: Callable[[], None] | None
    num_tokens: int
    mlx_uid: int | None = None
    start_time: float = field(default_factory=time.perf_counter)


def can_defer_prefill(
    group: mx.distributed.Group | None,
    has_prefix_cache: bool,
    use_prefix_cache: bool,
    has_vision: bool,
    is_bench: bool,
    is_ring: bool = False,
) -> bool:
    """Whether a prompt can skip eager prefill and use the chunked deferred path.

    TODO #7: the eager prefill in ``submit`` runs the WHOLE prompt synchronously
    on the GPU before inserting into the batch generator, which stalls the
    current batch's decode. The deferred path hands the full prompt to mlx-lm's
    BatchGenerator, which decodes the current batch first and prefills new
    prompts in chunks gated by batch capacity. Only safe when nothing needs the
    eager prefill's side effects: no distributed pipeline sync, no prefix-cache
    save/restore, no vision embedding patching, no bench timing and no
    ring sequence-parallel prefill.

    Ring prefill never returns True here (``is_ring`` always False for the
    mlx-lm deferred path) — ring prompts need distributed collectives that
    ``MlxBatchGenerator`` cannot drive, so they use the custom chunked
    ``_PendingRingInsert`` path instead.
    """
    return (
        group is None
        and (not has_prefix_cache or not use_prefix_cache)
        and not has_vision
        and not is_bench
    )


@dataclass(eq=False)
class ExoBatchGenerator:
    """Batcher multiplexing concurrent text-generation tasks over one model.

    Layers an exo task book-keeping layer (active tasks, prefix-cache
    persistence, token relay, per-task generation state) on top of
    ``MlxBatchGenerator`` so multiple requests share a single MLX decode
    loop without starving each other.
    """

    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None
    vision_processor: VisionProcessor | None = None

    _mlx_gen: MlxBatchGenerator = field(init=False)
    _active_tasks: dict[int, _EngineTask] = field(default_factory=dict, init=False)
    _supports_token_relay: bool = field(init=False)
    _pending_ring_inserts: dict[int, _PendingRingInsert] = field(
        default_factory=dict, init=False
    )
    _synthetic_uid: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._mlx_gen = MlxBatchGenerator(
            model=self.model,
            stop_tokens=[[t] for t in eos_ids_from_tokenizer(self.tokenizer)],
            prefill_step_size=4096,
        )
        self._step_count = 0
        self._supports_token_relay = self.group is not None and any(
            isinstance(layer, PipelineLastLayer) for layer in self.model.layers
        )

    @property
    def has_work(self) -> bool:
        """True if any active task or pending MLX batch remains."""
        return (
            bool(self._active_tasks)
            or bool(self._pending_ring_inserts)
            or bool(self._mlx_gen._unprocessed_sequences)
            or len(self._mlx_gen._prompt_batch) > 0
            or len(self._mlx_gen._generation_batch) > 0
        )

    def submit(
        self,
        task_params: TextGenerationTaskParams,
        prompt: str,
        on_prefill_progress: Callable[[int, int], None] | None = None,
        distributed_prompt_progress_callback: Callable[[], None] | None = None,
        on_generation_token: Callable[[], None] | None = None,
    ) -> int:
        """Submit a generation task; returns its uid for later ``step``/cancel."""
        all_prompt_tokens = encode_prompt(self.tokenizer, prompt)
        all_prompt_tokens = fix_unmatched_think_end_tokens(
            all_prompt_tokens, self.tokenizer
        )

        vision: VisionResult | None = None
        media_regions: list[MediaRegion] = []

        if self.vision_processor is not None:
            try:
                vision = prepare_vision(
                    images=task_params.images,
                    chat_template_messages=task_params.chat_template_messages,
                    vision_processor=self.vision_processor,
                    tokenizer=self.tokenizer,
                    model=self.model,
                    model_id=task_params.model,
                    task_params=task_params,
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Vision processing failed, falling back to text-only"
                )

        if vision is not None:
            all_prompt_tokens = vision.prompt_tokens
            media_regions = vision.media_regions

        is_bench = task_params.bench

        # TODO #7: when this prompt needs nothing beyond a plain KV cache
        # (single node, no prefix cache, no vision, no bench), defer prefill to
        # mlx-lm's BatchGenerator: it decodes the current batch FIRST and
        # prefills new prompts in chunks gated by batch capacity, so a long new
        # prompt never blocks the current batch's decode. The eager path below
        # (distributed pipeline sync, prefix-cache save, remote prefill) stays
        # for the cases that need it.
        defer_prefill = can_defer_prefill(
            group=self.group,
            has_prefix_cache=self.kv_prefix_cache is not None,
            use_prefix_cache=task_params.use_prefix_cache,
            has_vision=vision is not None,
            is_bench=task_params.bench,
        )

        prefix_hit_length = 0
        matched_index: int | None = None
        is_exact_hit = False
        prompt_tokens = all_prompt_tokens

        if self.kv_prefix_cache is not None and task_params.use_prefix_cache:
            # Prefix entries are persistent allocations. Reclaim enough of them
            # for prefill's temporary activations before copying a cache hit.
            self.kv_prefix_cache.evict_for_prefill()
            cache, remaining_tokens, matched_index, is_exact_hit = (
                self.kv_prefix_cache.get_kv_cache(
                    self.model, all_prompt_tokens, media_regions=media_regions
                )
            )
            prefix_hit_length = len(all_prompt_tokens) - len(remaining_tokens)
            if not mx_ranks_agree_on_value(prefix_hit_length, self.group):
                # Divergent restore positions would make ranks prefill
                # different token counts and deadlock the pipeline.
                logger.warning(
                    "KV prefix cache hit lengths diverge across pipeline ranks; "
                    "discarding the hit to keep prefill in lockstep"
                )
                cache = make_kv_cache(
                    self.model,
                    max_kv_size=MAX_KV_SIZE,
                    keep=KEEP_KV_SIZE or 0,
                )
                prefix_hit_length = 0
                matched_index = None
                is_exact_hit = False
            elif prefix_hit_length > 0:
                logger.info(
                    f"KV cache hit: {prefix_hit_length}/{len(all_prompt_tokens)} tokens "
                    f"cached ({100 * prefix_hit_length / len(all_prompt_tokens):.1f}%)"
                )
                prompt_tokens = remaining_tokens
        else:
            cache = make_kv_cache(
                self.model, max_kv_size=MAX_KV_SIZE, keep=KEEP_KV_SIZE or 0
            )

        seed = task_params.seed if task_params.seed is not None else 42
        mx.random.seed(seed)

        sampler = make_sampler(
            temp=task_params.temperature
            if task_params.temperature is not None
            else 0.7,
            top_p=task_params.top_p if task_params.top_p is not None else 1.0,
            min_p=task_params.min_p if task_params.min_p is not None else 0.05,
            top_k=task_params.top_k if task_params.top_k is not None else 0,
        )

        vision_ctx = (
            patch_embed_tokens(
                self.model,
                vision.embeddings,
                prefix_hit_length,
                len(prompt_tokens) - 1,
                image_token_id=vision.image_token_id,
            )
            if vision is not None
            else contextlib.nullcontext()
        )
        uncached_count = len(prompt_tokens)
        is_ring = uses_ring_sequence_parallel_prefill(
            self.model, len(prompt_tokens) - 1, self.group
        )
        use_remote = (
            uncached_count > REMOTE_PREFILL_MIN_TOKENS
            and task_params.prefill_endpoint is not None
            and not is_ring
        )

        # Ring sequence-parallel prefill runs through distributed collectives
        # that MlxBatchGenerator's PromptProcessingBatch cannot drive.  Instead
        # of the synchronous eager prefill below (which blocks submit() and
        # stalls the current batch's decode for the whole prompt), defer it:
        # build a chunked RingPrefillState now, return a synthetic uid, and
        # drain one chunk per step() until done, then insert the decode seed.
        ring_state: RingPrefillState | None = None
        if is_ring and not use_remote:
            assert self.group is not None
            ring_state = make_ring_prefill_state(
                self.model, prompt_tokens[:-1], cache, self.group
            )

        _prefill_tps: float = 0.0
        _prefill_tokens: int = 0
        cache_snapshots: list[CacheSnapshot] = []
        remote_prefilled = False
        if not defer_prefill and ring_state is None:
            with vision_ctx:
                if use_remote and task_params.prefill_endpoint is not None:
                    try:
                        _prefill_tps, _prefill_tokens, cache_snapshots = remote_prefill(
                            prompt_tokens[:-1],
                            cache,
                            on_prefill_progress,
                            endpoint=task_params.prefill_endpoint,
                            request_id=str(uuid.uuid4()),
                            model_id=str(task_params.model),
                            start_pos=prefix_hit_length,
                        )
                        remote_prefilled = True
                    except Exception:
                        logger.opt(exception=True).warning(
                            "Remote prefill failed, falling back to local prefill"
                        )

                if not remote_prefilled:
                    _prefill_tps, _prefill_tokens, cache_snapshots = prefill(
                        self.model,
                        self.tokenizer,
                        sampler,
                        prompt_tokens[:-1],
                        cache,
                        self.group,
                        on_prefill_progress,
                        distributed_prompt_progress_callback,
                    )

        prefix_cache_hit: Literal["none", "partial", "exact"] = "none"
        if matched_index is not None and prefix_hit_length > 0:
            assert self.kv_prefix_cache is not None
            if is_exact_hit:
                prefix_cache_hit = "exact"
                _prefill_tps = self.kv_prefix_cache.prefill_tps[matched_index]
            else:
                prefix_cache_hit = "partial"

        # We need to clamp rotating kv caches to max size so that mlx lm's _merge_caches behaves
        for c in cache:
            if (
                isinstance(c, RotatingKVCache)
                and c.keys is not None
                and c.values is not None
                and c.keys.shape[2] > c.max_size
            ):
                trim_size = c.keys.shape[2] - c.max_size
                c.keys = c._trim(trim_size, c.keys)
                c.values = c._trim(trim_size, c.values)
                c._idx = c.max_size

        if task_params.use_prefix_cache and not defer_prefill and ring_state is None:
            min_prefix_hit_length = max(
                1000, system_prompt_token_count(task_params, self.tokenizer)
            )
            self._save_prefix_cache(
                all_prompt_tokens,
                list(cache),
                cache_snapshots,
                prefix_hit_length,
                matched_index,
                min_prefix_hit_length,
                media_regions,
                prefill_tps=_prefill_tps,
            )

        last_tokens = (
            prompt_tokens[-1:]
            if uses_ring_sequence_parallel_prefill(
                self.model, len(prompt_tokens) - 1, self.group
            )
            else prompt_tokens[-2:]
        )

        logits_processors: list[Callable[[mx.array, mx.array], mx.array]] = (
            make_logits_processors(
                logit_bias=task_params.logit_bias,
                repetition_penalty=task_params.repetition_penalty,
                repetition_context_size=task_params.repetition_context_size
                if task_params.repetition_context_size is not None
                else 20,
                presence_penalty=task_params.presence_penalty,
                frequency_penalty=task_params.frequency_penalty,
            )
        )
        if is_bench:
            # Only sample length eos tokens
            eos_ids = eos_ids_from_tokenizer(self.tokenizer)
            logits_processors = [ban_token_ids(eos_ids)] + logits_processors

        # T28: JSON-schema constrained decoding (fail-open on unsupported schemas).
        constrained = make_constrained_processor(task_params, self.tokenizer)
        if constrained is not None:
            logits_processors = [constrained] + logits_processors

        max_tokens = task_params.max_output_tokens or MAX_TOKENS

        if ring_state is not None:
            # Deferred ring prefill: build the pending-insert record now, return
            # a synthetic uid, and let step() drain ring chunks before finally
            # inserting the decode seed with the fully-populated cache.
            synthetic_uid = -(self._synthetic_uid + 1)
            self._synthetic_uid += 1
            self._pending_ring_inserts[synthetic_uid] = _PendingRingInsert(
                uid=synthetic_uid,
                ring_state=ring_state,
                last_tokens=(
                    prompt_tokens[-1:]
                    if is_ring
                    else prompt_tokens[-2:]
                ),
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                cache=cache,
                on_prefill_progress=on_prefill_progress,
                distributed_prompt_progress_callback=(
                    distributed_prompt_progress_callback
                ),
                num_tokens=len(prompt_tokens) - 1,
            )
            uid = synthetic_uid
        else:
            if defer_prefill:
                # Deferred path: hand the FULL prompt to the BatchGenerator with an
                # empty cache — its chunked PromptProcessingBatch prefills it in
                # prefill_step_size chunks, decode-first, gated by batch capacity.
                insert_tokens = all_prompt_tokens
                insert_caches: list[list[Any] | None] = [None]
            else:
                last_tokens = (
                    prompt_tokens[-1:]
                    if uses_ring_sequence_parallel_prefill(
                        self.model, len(prompt_tokens) - 1, self.group
                    )
                    else prompt_tokens[-2:]
                )
                insert_tokens = last_tokens
                insert_caches = [list(cache)]

            uids = self._mlx_gen.insert(
                prompts=[cast(list[int], insert_tokens.tolist())],
                max_tokens=[max_tokens],
                caches=cast(list[list[Any]] | None, insert_caches),
                samplers=[sampler],
                logits_processors=[logits_processors],
            )

            assert len(uids) == 1

            uid = uids[0]

        self._active_tasks[uid] = _EngineTask(
            uid=uid,
            task_params=task_params,
            all_prompt_tokens=all_prompt_tokens,
            prefix_hit_length=prefix_hit_length,
            matched_index=matched_index,
            detokenizer=self.tokenizer.detokenizer,
            on_generation_token=on_generation_token,
            generation_start_time=time.perf_counter(),
            prefill_tps=_prefill_tps,
            prefix_cache_hit=prefix_cache_hit,
            media_regions=media_regions,
        )

        return uid

    def step(self) -> list[tuple[int, GenerationResponse]]:
        """Advance the decode loop one step, returning new token responses."""
        if not self.has_work:
            # Idle safe point: run the periodic KV cleanup (time-gated,
            # no-op within the interval) so a node sitting idle returns
            # stale Metal buffers instead of pinning them until the next
            # eviction/threshold crossing.
            if self.kv_prefix_cache is not None:
                try:
                    self.kv_prefix_cache._periodic_cleanup()
                except Exception:
                    logger.warning(
                        "Periodic KV cleanup failed", exc_info=True
                    )
            return []

        # Interleave ring prefill with decode: run ONE chunk per pending ring
        # task per step() call so a long ring prompt no longer blocks decode.
        # When a task's chunks are exhausted, insert its decode seed into
        # MlxBatchGenerator with the now-populated cache and remap its uid.
        if self._pending_ring_inserts:
            self._drain_ring_prefill_chunks()

        gb = self._mlx_gen._generation_batch
        needs_logprobs = any(
            t.task_params.logprobs for t in self._active_tasks.values()
        )
        set_needs_topk(gb, needs_logprobs)

        # Token relay needs every rank's logits path untouched only on the last
        # rank; requests that ask for logprobs need real logits on rank 0, so
        # fall back to the legacy all_gather decode for those. The flag is
        # reset after the step so prefill and warmup always use legacy paths.
        set_pipeline_token_relay(
            self.model, self._supports_token_relay and not needs_logprobs
        )
        _step_tic = time.perf_counter()
        try:
            _, responses = self._mlx_gen.next()
        finally:
            set_pipeline_token_relay(self.model, False)
        _next_elapsed = time.perf_counter() - _step_tic

        topk = take_ready_topk(gb)

        results: list[tuple[int, GenerationResponse]] = []

        for response in responses:
            if response.uid not in self._active_tasks:
                logger.warning(
                    f"response uid {response.uid} was not found - should be active"
                )
                continue

            state = self._active_tasks[response.uid]
            now = time.perf_counter()
            if state.first_gen_token_time is None:
                state.first_gen_token_time = now
            state.last_gen_token_time = now
            if state.on_generation_token is not None:
                state.on_generation_token()
            if response.finish_reason != "stop":
                state.detokenizer.add_token(response.token)
            if response.finish_reason is not None:
                state.detokenizer.finalize()
            # last_segment is the DELTA since the previous access (the base
            # StreamingDetokenizer property advances its offset on every
            # read). Emit it as-is — like the sequential path's out.text.
            text = state.detokenizer.last_segment
            state.completion_tokens += 1
            if state.task_params.bench:
                delta = now - state.first_gen_token_time
                logger.debug(
                    f"[bench] uid={response.uid} tok#{state.completion_tokens} {text!r} t={delta:.4f}s"
                )
            state.generated_text_parts.append(text)

            # Hold back any trailing partial stop-sequence match so a multi-token
            # stop sequence never leaks its leading bytes into output.
            model_finish_reason = cast(FinishReason | None, response.finish_reason)
            task_params = state.task_params
            stop_sequences = _stop_sequences(task_params)

            state.potential_stop_sequence_text += text
            text, matched_stop_sequence, state.potential_stop_sequence_text = (
                scan_stop_sequences(state.potential_stop_sequence_text, stop_sequences)
            )

            finish_reason: FinishReason | None
            if matched_stop_sequence is not None:
                finish_reason = "stop"
            elif model_finish_reason is not None:
                # Natural EOS / length limit: flush any held-back text — it is
                # real output that merely looked like a stop-sequence prefix.
                text += state.potential_stop_sequence_text
                state.potential_stop_sequence_text = ""
                finish_reason = model_finish_reason
            else:
                finish_reason = None

            is_done = finish_reason is not None

            logprob: float | None = None
            top_logprobs: list[TopLogprobItem] | None = None
            if task_params.logprobs:
                precomputed = topk.for_uid(response.uid)
                precomputed_indices, precomputed_values, precomputed_selected = (
                    precomputed if precomputed is not None else (None, None, None)
                )
                with mx.stream(generation_stream):
                    logprob, top_logprobs = extract_top_logprobs(
                        logprobs=response.logprobs,
                        tokenizer=self.tokenizer,
                        top_logprobs=task_params.top_logprobs or DEFAULT_TOP_LOGPROBS,
                        selected_token=response.token,
                        precomputed_indices=precomputed_indices,
                        precomputed_values=precomputed_values,
                        precomputed_selected=precomputed_selected,
                    )

            stats: GenerationStats | None = None
            usage: Usage | None = None
            if is_done:
                if state.completion_tokens > 1:
                    gen_span = state.last_gen_token_time - state.first_gen_token_time
                    generation_tps = (
                        (state.completion_tokens - 1) / gen_span
                        if gen_span > 0
                        else 0.0
                    )
                else:
                    generation_tps = 0.0

                stats = GenerationStats(
                    prompt_tps=state.prefill_tps,
                    generation_tps=generation_tps,
                    prompt_tokens=len(state.all_prompt_tokens),
                    generation_tokens=state.completion_tokens,
                    peak_memory_usage=Memory.from_gb(mx.get_peak_memory() / 1e9),
                    prefix_cache_hit=state.prefix_cache_hit,
                )
                total_prompt_tokens = len(state.all_prompt_tokens)
                usage = Usage(
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=state.completion_tokens,
                    total_tokens=total_prompt_tokens + state.completion_tokens,
                    prompt_tokens_details=PromptTokensDetails(
                        cached_tokens=state.prefix_hit_length
                    ),
                    completion_tokens_details=CompletionTokensDetails(
                        reasoning_tokens=0
                    ),
                )

            results.append(
                (
                    response.uid,
                    GenerationResponse(
                        text=text,
                        token=response.token,
                        logprob=logprob,
                        top_logprobs=top_logprobs,
                        finish_reason=finish_reason,
                        stats=stats,
                        usage=usage,
                        matched_stop_sequence=matched_stop_sequence,
                    ),
                )
            )

            if is_done:
                del self._active_tasks[response.uid]

        _step_elapsed = time.perf_counter() - _step_tic
        _overhead = _step_elapsed - _next_elapsed
        self._step_count += 1
        if self._step_count % 64 == 0 and responses:
            logger.debug(
                f"step overhead: {_overhead * 1000:.2f}ms (next={_next_elapsed * 1000:.2f}ms total={_step_elapsed * 1000:.2f}ms)"
            )

        return results

    def _drain_ring_prefill_chunks(self) -> None:
        """Run one ring-prefill chunk per pending task; insert seeds when done.

        Called at the top of each ``step()``.  The chunked ``RingPrefillState``
        drives the distributed ring collectives; the ``MlxBatchGenerator`` is
        handed only the *last* token (decode seed) once the KV cache is fully
        populated, exactly like the eager path, so its own (non-ring-aware)
        prompt processing never touches the distributed cache.
        """
        for synthetic_uid, pending in list(self._pending_ring_inserts.items()):
            if pending.ring_state.step():
                # One chunk processed. Fire progress callbacks (same contract
                # as eager prefill: chunk_end is a global token index).
                if pending.on_prefill_progress is not None:
                    pending.on_prefill_progress(
                        pending.ring_state.index, pending.num_tokens
                    )
                if pending.distributed_prompt_progress_callback is not None:
                    pending.distributed_prompt_progress_callback()
                return
            # Chunks exhausted: insert the decode seed with the populated cache.
            mlx_uids = self._mlx_gen.insert(
                prompts=[cast(list[int], pending.last_tokens.tolist())],
                max_tokens=[pending.max_tokens],
                caches=cast(list[list[Any]] | None, [list(pending.cache)]),
                samplers=[pending.sampler],
                logits_processors=[pending.logits_processors],
            )
            del self._pending_ring_inserts[synthetic_uid]
            if not mlx_uids:
                logger.warning(
                    f"ring prefill uid {synthetic_uid}: no mlx uid returned; task dropped"
                )
                self._active_tasks.pop(synthetic_uid, None)
                continue
            pending.mlx_uid = mlx_uids[0]
            assert pending.mlx_uid is not None
            # Remap the engine task from synthetic uid to real mlx uid so the
            # response loop below and cancel() see a consistent uid namespace.
            engine_task = self._active_tasks.pop(synthetic_uid, None)
            if engine_task is not None:
                engine_task.uid = pending.mlx_uid
                elapsed = time.perf_counter() - pending.start_time
                engine_task.prefill_tps = (
                    pending.num_tokens / elapsed if elapsed > 0 else 0.0
                )
                self._active_tasks[pending.mlx_uid] = engine_task

    def cancel(self, uids: list[int]) -> None:
        """Cancel the given task uids, removing them from the batch and bookkeeping."""
        # Pending ring tasks have a synthetic uid and are not yet in the mlx
        # batch; drop them from bookkeeping directly.
        for uid in uids:
            pending = self._pending_ring_inserts.pop(uid, None)
            if pending is not None:
                self._active_tasks.pop(uid, None)
                continue
        # Synthetic uids are negative; mlx uids are non-negative, so filter
        # only the real ones for mlx remove().
        mlx_uids = [u for u in uids if u >= 0]
        if mlx_uids:
            self._mlx_gen.remove(mlx_uids)
        for uid in uids:
            self._active_tasks.pop(uid, None)

    def close(self) -> None:
        """Shut down the underlying batch generator and clear the MLX cache."""
        self._mlx_gen.close()
        for pending in self._pending_ring_inserts.values():
            if pending.ring_state.stall_watchdog is not None:
                pending.ring_state.stall_watchdog.close()
        self._pending_ring_inserts.clear()
        mx.clear_cache()

    def _save_prefix_cache(
        self,
        all_prompt_tokens: mx.array,
        cache: KVCacheType,
        cache_snapshots: list[CacheSnapshot] | None,
        prefix_hit_length: int,
        matched_index: int | None,
        min_prefix_hit_length: int = 1000,
        media_regions: list[MediaRegion] | None = None,
        prefill_tps: float = 0.0,
    ) -> None:
        if self.kv_prefix_cache is None:
            return

        try:
            if self.kv_prefix_cache.should_update_entry(
                matched_index, prefix_hit_length, min_prefix_hit_length
            ):
                assert matched_index is not None
                self.kv_prefix_cache.update_kv_cache(
                    matched_index,
                    all_prompt_tokens,
                    cache,
                    cache_snapshots,
                    restore_pos=prefix_hit_length,
                    media_regions=media_regions,
                    prefill_tps=prefill_tps,
                )
            else:
                self.kv_prefix_cache.add_kv_cache(
                    all_prompt_tokens,
                    cache,
                    cache_snapshots,
                    media_regions=media_regions,
                    prefill_tps=prefill_tps,
                )
        except Exception:
            logger.warning("Failed to save prefix cache", exc_info=True)
