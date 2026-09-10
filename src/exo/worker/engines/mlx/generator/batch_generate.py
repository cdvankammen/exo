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
from exo.worker.engines.mlx.generator.chunked_ring_prefill import (
    ChunkedRingPrefill,
    PrefillCancelled,
)
from exo.worker.engines.mlx.generator.generate import (
    ban_token_ids,
    eos_ids_from_tokenizer,
    extract_top_logprobs,
    make_constrained_processor,
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

    # For ring-chunked prefill: the task is submitted but not yet inserted
    # into the mlx batch.  We stash the state needed for final insertion.
    ring_chunked_cache: KVCacheType | None = None
    ring_chunked_prompt_tokens: mx.array | None = None
    ring_chunked_sampler: Callable[[mx.array], mx.array] | None = None
    ring_chunked_logits_processors: list[Callable[[mx.array, mx.array], mx.array]] | None = None
    ring_chunked_max_tokens: int | None = None
    ring_chunked_task_params: TextGenerationTaskParams | None = None


def can_defer_prefill(
    group: mx.distributed.Group | None,
    has_prefix_cache: bool,
    use_prefix_cache: bool,
    has_vision: bool,
    is_bench: bool,
) -> bool:
    """Whether a prompt can skip eager prefill and use the chunked deferred path.

    TODO #7: the eager prefill in ``submit`` runs the WHOLE prompt synchronously
    on the GPU before inserting into the batch generator, which stalls the
    current batch's decode. The deferred path hands the full prompt to mlx-lm's
    BatchGenerator, which decodes the current batch first and prefills new
    prompts in chunks gated by batch capacity. Only safe when nothing needs the
    eager prefill's side effects: no distributed pipeline sync, no prefix-cache
    save/restore, no vision embedding patching, no bench timing.
    """
    return (
        group is None
        and (not has_prefix_cache or not use_prefix_cache)
        and not has_vision
        and not is_bench
    )


@dataclass(eq=False)
class ExoBatchGenerator:
    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None
    vision_processor: VisionProcessor | None = None

    _mlx_gen: MlxBatchGenerator = field(init=False)
    _active_tasks: dict[int, _EngineTask] = field(default_factory=dict, init=False)
    _supports_token_relay: bool = field(init=False)
    _chunked_ring_prefill: ChunkedRingPrefill | None = field(default=None, init=False)
    _ring_chunked_uid: int | None = field(default=None, init=False)

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
        return (
            bool(self._active_tasks)
            or self._chunked_ring_prefill is not None
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

        # Evict cached prefixes BEFORE prefill regardless of prefix cache
        # settings.  Prefill's forward pass needs temporary activation memory
        # in addition to the persistent KV cache.  Without pre-prefill
        # eviction the system can OOM when cached entries consume most of
        # available memory.  (#2182)
        if self.kv_prefix_cache is not None:
            self.kv_prefix_cache.evict_for_prefill()

        if self.kv_prefix_cache is not None and task_params.use_prefix_cache:
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
        use_remote = (
            uncached_count > REMOTE_PREFILL_MIN_TOKENS
            and task_params.prefill_endpoint is not None
            and not uses_ring_sequence_parallel_prefill(
                self.model, len(prompt_tokens) - 1, self.group
            )
        )

        _prefill_tps: float = 0.0
        _prefill_tokens: int = 0
        cache_snapshots: list[CacheSnapshot] = []
        remote_prefilled = False

        # Ring-eligible prompts must NOT use the mlx deferred path (which
        # feeds identical tokens to every rank) or the eager synchronous path
        # (which blocks decode for the whole prompt).  Instead they get a
        # ChunkedRingPrefill state machine that advances one chunk per batch
        # scheduler tick, letting decode interleave between chunks.
        is_ring_chunked = (
            not defer_prefill
            and self.group is not None
            and uses_ring_sequence_parallel_prefill(
                self.model, len(prompt_tokens) - 1, self.group
            )
        )

        if is_ring_chunked:
            assert self.group is not None
            assert self._chunked_ring_prefill is None, (
                "A chunked ring prefill is already in flight"
            )
            self._chunked_ring_prefill = ChunkedRingPrefill(
                prompt_tokens=prompt_tokens[:-1],
                cache=cache,
                model=self.model,
                group=self.group,
                prefix_hit_length=prefix_hit_length,
                on_prefill_progress=on_prefill_progress,
                distributed_prompt_progress_callback=(
                    distributed_prompt_progress_callback
                ),
            )
            logger.info(
                f"Deferred ring prefill to chunked path for "
                f"{len(prompt_tokens) - 1} tokens"
            )
        elif not defer_prefill:
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

        if task_params.use_prefix_cache and not defer_prefill:
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

        if is_ring_chunked:
            # Chunked ring prefill: skip the mlx insert.  The KV cache is
            # filled by ChunkedRingPrefill.advance_chunk() in step().  The
            # last token is inserted only after all chunks complete.
            uid = self._mlx_gen._uid_count
            self._mlx_gen._uid_count += 1

            self._ring_chunked_uid = uid
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
                ring_chunked_cache=list(cache),
                ring_chunked_prompt_tokens=prompt_tokens,
                ring_chunked_sampler=sampler,
                ring_chunked_logits_processors=logits_processors,
                ring_chunked_max_tokens=max_tokens,
                ring_chunked_task_params=task_params,
            )
            return uid

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
        if not self.has_work:
            return []

        # Advance chunked ring prefill if in flight.  Each step processes
        # exactly one chunk, then the model returns to normal (non-ring)
        # mode so decode can run for other active tasks between chunks.
        if self._chunked_ring_prefill is not None:
            try:
                self._chunked_ring_prefill.advance_chunk()
            except PrefillCancelled:
                # Distributed cancellation arrived during the chunk.
                if self._ring_chunked_uid is not None:
                    self._active_tasks.pop(self._ring_chunked_uid, None)
                self._chunked_ring_prefill = None
                self._ring_chunked_uid = None
                return []

            if self._chunked_ring_prefill.all_done:
                # All chunks done: finalise and insert the last token into
                # the mlx batch so decode can start.
                assert self._ring_chunked_uid is not None
                uid = self._ring_chunked_uid
                state = self._active_tasks.get(uid)
                if state is not None:
                    _prefill_tps, _prefill_tokens = self._chunked_ring_prefill.finish(
                        prefix_cache_save_fn=(
                            self._save_prefix_cache
                            if state.task_params.use_prefix_cache
                            else None
                        ),
                    )
                    state.prefill_tps = _prefill_tps

                    # Now insert the last token with the filled cache.
                    assert state.ring_chunked_prompt_tokens is not None
                    assert state.ring_chunked_cache is not None
                    assert state.ring_chunked_sampler is not None
                    assert state.ring_chunked_logits_processors is not None
                    assert state.ring_chunked_max_tokens is not None

                    last_tokens = state.ring_chunked_prompt_tokens[-1:]
                    uids = self._mlx_gen.insert(
                        prompts=[cast(list[int], last_tokens.tolist())],
                        max_tokens=[state.ring_chunked_max_tokens],
                        caches=cast(
                            list[list[Any]] | None,
                            [state.ring_chunked_cache],
                        ),
                        samplers=[state.ring_chunked_sampler],
                        logits_processors=[state.ring_chunked_logits_processors],
                    )
                    assert len(uids) == 1
                    # The mlx_gen may assign a different uid; remap.
                    new_uid = uids[0]
                    if new_uid != uid:
                        self._active_tasks[new_uid] = self._active_tasks.pop(uid)
                        self._active_tasks[new_uid].uid = new_uid

                self._chunked_ring_prefill = None
                self._ring_chunked_uid = None

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

                gen_span_ms = (
                    (state.last_gen_token_time - state.first_gen_token_time) * 1000
                    if state.first_gen_token_time is not None
                    and state.last_gen_token_time is not None
                    and state.completion_tokens > 1
                    else None
                )
                stats = GenerationStats(
                    prompt_tps=state.prefill_tps,
                    generation_tps=generation_tps,
                    prompt_tokens=len(state.all_prompt_tokens),
                    generation_tokens=state.completion_tokens,
                    peak_memory_usage=Memory.from_gb(mx.get_peak_memory() / 1e9),
                    prefix_cache_hit=state.prefix_cache_hit,
                    decode_latency_ms=gen_span_ms,
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

    def cancel(self, uids: list[int]) -> None:
        self._mlx_gen.remove(uids)
        for uid in uids:
            self._active_tasks.pop(uid, None)

        # If the ring-chunked request was cancelled mid-prefill, abort it.
        if self._ring_chunked_uid is not None and self._ring_chunked_uid in set(uids):
            if self._chunked_ring_prefill is not None:
                if self._chunked_ring_prefill._watchdog is not None:
                    self._chunked_ring_prefill._watchdog.close()
            self._chunked_ring_prefill = None
            self._ring_chunked_uid = None

    def close(self) -> None:
        self._mlx_gen.close()
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
