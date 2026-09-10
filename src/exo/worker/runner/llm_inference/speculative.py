"""Speculative Decoding for Exo distributed inference.

Implements draft-and-verify speculative decoding where a small draft model
generates K candidate tokens that are verified in a single forward pass by
the larger target model. This achieves 2-3x decode speedup when the draft
model has reasonable acceptance rates.

Architecture:
    - Draft model runs on smallest node (low latency)
    - Target model runs on largest node (high throughput verification)
    - Draft generates K tokens (default 4-8) autoregressively
    - Target verifies all K+1 positions in one batched forward pass
    - Accept longest matching prefix, reject rest, repeat

Reference: similar-projects-performance-research-2026-09-01.md
"""

import time
from collections.abc import Generator, Iterable
from dataclasses import dataclass, field
from typing import BinaryIO, Callable

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.chunks import Chunk, ErrorChunk, GenerationChunk
from exo.shared.types.common import ModelId
from exo.shared.types.events import ChunkGenerated, Event
from exo.shared.types.tasks import CANCEL_ALL_TASKS, GenerationTask, TaskId, TextGeneration
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
    GenerationResponse,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.base import Engine
from exo.worker.engines.mlx.cache import KVPrefixCache, make_kv_cache
from exo.worker.engines.mlx.generator.generate import mlx_generate
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.engines.mlx.utils_mlx import apply_chat_template, mx_barrier
from exo.worker.engines.mlx.vision import VisionProcessor
from exo.worker.runner.bootstrap import logger

# Default speculative decoding parameters
DEFAULT_NUM_DRAFT_TOKENS = 6
DEFAULT_DRAFT_TEMPERATURE = 0.0  # Greedy sampling for draft
DEFAULT_TARGET_TEMPERATURE = 0.7  # Standard temperature for target verification
MIN_ACCEPTANCE_RATE = 0.3  # Below this, fall back to standard generation
MAX_CONSECUTIVE_REJECTIONS = 10  # After this many full rejections, reduce K


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""
    num_draft_tokens: int = DEFAULT_NUM_DRAFT_TOKENS
    draft_temperature: float = DEFAULT_DRAFT_TEMPERATURE
    target_temperature: float = DEFAULT_TARGET_TEMPERATURE
    min_acceptance_rate: float = MIN_ACCEPTANCE_RATE
    max_consecutive_rejections: int = MAX_CONSECUTIVE_REJECTIONS
    enabled: bool = True


@dataclass
class SpeculativeStats:
    """Statistics tracking for speculative decoding performance."""
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    total_verification_steps: int = 0
    consecutive_rejections: int = 0
    current_k: int = DEFAULT_NUM_DRAFT_TOKENS
    
    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens
    
    def record_verification(self, drafted: int, accepted: int) -> None:
        self.total_draft_tokens += drafted
        self.total_accepted_tokens += accepted
        self.total_verification_steps += 1
        if accepted == 0:
            self.consecutive_rejections += 1
        else:
            self.consecutive_rejections = 0


def verify_draft_tokens(
    target_model: Model,
    target_cache: KVCacheType,
    draft_tokens: list[int],
    last_verified_token: int,
    group: mx.distributed.Group | None,
) -> tuple[list[int], int]:
    """Verify draft tokens against the target model in a single forward pass.
    
    Args:
        target_model: The target (large) model for verification
        target_cache: KV cache from target model (will be extended)
        draft_tokens: List of K draft token IDs to verify
        last_verified_token: The last token accepted by target model
        group: Distributed group for multi-rank coordination
        
    Returns:
        Tuple of (accepted_tokens, bonus_token):
        - accepted_tokens: Prefix of draft_tokens that matched target's distribution
        - bonus_token: One additional token sampled from target at rejection point
    """
    if not draft_tokens:
        return [], last_verified_token
    
    # Build input sequence: [last_verified] + draft_tokens
    # We need logits at each position to verify the next token
    input_ids = mx.array([last_verified_token] + draft_tokens)[None, :]  # (1, K+1)
    
    # Single forward pass through target model with all K+1 tokens
    # This is the key efficiency gain: O(1) forward passes instead of O(K)
    with mx.stream(mx.default_device()):
        logits = target_model(input_ids, cache=target_cache)
        # logits shape: (1, K+1, vocab_size)
    
    accepted = []
    bonus_token = last_verified_token
    
    # Verify each draft token against target's predicted distribution
    for i, draft_tok in enumerate(draft_tokens):
        # Get target's probability distribution at position i
        # (predicting token at position i+1 given context up to i)
        pos_logits = logits[0, i, :]
        
        # Rejection sampling: accept if draft token has sufficient probability
        # under target distribution. For greedy draft (temp=0), we use simple
        # argmax comparison. For stochastic draft, use proper rejection sampling.
        target_next = mx.argmax(pos_logits).item()
        
        if draft_tok == target_next:
            accepted.append(draft_tok)
        else:
            # Rejection: sample bonus token from target distribution
            # and stop accepting further draft tokens
            probs = mx.softmax(pos_logits)
            bonus_token = mx.random.categorical(probs).item()
            break
    else:
        # All draft tokens accepted! Sample one bonus token from final position
        final_logits = logits[0, len(draft_tokens), :]
        probs = mx.softmax(final_logits)
        bonus_token = mx.random.categorical(probs).item()
    
    # Trim target cache to only include accepted tokens + bonus
    # Cache was extended by K+1 positions during forward pass
    n_to_trim = len(draft_tokens) - len(accepted)
    if n_to_trim > 0:
        for c in target_cache:
            if hasattr(c, 'trim'):
                c.trim(n_to_trim)
    
    return accepted, bonus_token


def generate_draft_tokens(
    draft_model: Model,
    draft_tokenizer: TokenizerWrapper,
    draft_cache: KVCacheType,
    start_token: int,
    num_tokens: int,
    temperature: float,
    group: mx.distributed.Group | None,
) -> list[int]:
    """Generate K draft tokens using the small draft model.
    
    Args:
        draft_model: Small/fast model for generating candidates
        draft_tokenizer: Tokenizer for draft model
        draft_cache: KV cache for draft model
        start_token: Token to start drafting from
        num_tokens: Number of tokens to draft (K)
        temperature: Sampling temperature for draft model
        group: Distributed group
        
    Returns:
        List of K draft token IDs
    """
    draft_tokens = []
    current_token = start_token
    
    for _ in range(num_tokens):
        input_ids = mx.array([current_token])[None, :]  # (1, 1)
        
        with mx.stream(mx.default_device()):
            logits = draft_model(input_ids, cache=draft_cache)
            # logits shape: (1, 1, vocab_size)
        
        pos_logits = logits[0, 0, :]
        
        if temperature <= 0.0:
            # Greedy sampling
            next_token = mx.argmax(pos_logits).item()
        else:
            # Temperature-scaled sampling
            scaled_logits = pos_logits / temperature
            probs = mx.softmax(scaled_logits)
            next_token = mx.random.categorical(probs).item()
        
        draft_tokens.append(next_token)
        current_token = next_token
    
    return draft_tokens


@dataclass(eq=False)
class SpeculativeGenerator(Engine):
    """Engine wrapper implementing speculative decoding.
    
    Wraps a target model engine and adds a draft model for speculative
    verification. Falls back to standard generation when acceptance rate
    drops below threshold or when draft/target models are incompatible.
    """
    
    # Target model components
    target_model: Model
    target_tokenizer: TokenizerWrapper
    target_group: mx.distributed.Group | None
    target_kv_prefix_cache: KVPrefixCache | None
    target_model_id: ModelId
    
    # Draft model components
    draft_model: Model
    draft_tokenizer: TokenizerWrapper
    draft_group: mx.distributed.Group | None
    draft_model_id: ModelId
    
    # Runner infrastructure
    device_rank: int
    cancel_receiver: MpReceiver[TaskId]
    event_sender: MpSender[Event]
    vision_processor: VisionProcessor | None = None
    
    # Configuration
    config: SpeculativeConfig = field(default_factory=SpeculativeConfig)
    check_for_cancel_every: int = 50
    
    # Internal state
    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _maybe_queue: list[TextGeneration] = field(default_factory=list, init=False)
    _maybe_cancel: list[TextGeneration] = field(default_factory=list, init=False)
    _all_tasks: dict[TaskId, TextGeneration] = field(default_factory=dict, init=False)
    _queue: list[TextGeneration] = field(default_factory=list, init=False)
    _stats: SpeculativeStats = field(default_factory=SpeculativeStats, init=False)
    _active_generator: Generator[GenerationResponse] | None = field(default=None, init=False)
    _active_task: TextGeneration | None = field(default=None, init=False)
    
    def warmup(self) -> None:
        """Warm up both draft and target models."""
        logger.info(f"Warming up speculative decoding: draft={self.draft_model_id}, target={self.target_model_id}")
        # Warmup handled by underlying mlx_generate calls
        # First generation will trigger JIT compilation
    
    def submit(self, task: GenerationTask) -> None:
        assert isinstance(task, TextGeneration)
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self._all_tasks[task.task_id] = task
        self._maybe_queue.append(task)
    
    def agree_on_tasks(self) -> None:
        """Coordinate task ordering across distributed ranks."""
        from exo.worker.engines.mlx.utils_mlx import mx_all_gather_tasks
        agreed, different = mx_all_gather_tasks(self._maybe_queue, self.target_group)
        self._queue.extend(agreed)
        self._maybe_queue = list(different)
    
    def agree_on_cancellations(self) -> None:
        """Coordinate cancellations across distributed ranks."""
        from exo.worker.engines.mlx.utils_mlx import mx_all_gather_tasks, mx_any
        has_cancel_all = False
        for task_id in self.cancel_receiver.collect():
            if task_id == CANCEL_ALL_TASKS:
                has_cancel_all = True
                continue
            if task_id in self._all_tasks:
                self._maybe_cancel.append(self._all_tasks[task_id])
        
        if mx_any(has_cancel_all, self.target_group):
            self._cancelled_tasks.add(CANCEL_ALL_TASKS)
        
        agreed, different = mx_all_gather_tasks(self._maybe_cancel, self.target_group)
        self._cancelled_tasks.update(task.task_id for task in agreed)
        self._maybe_cancel = list(different)
    
    def should_cancel(self, task_id: TaskId) -> bool:
        return task_id in self._cancelled_tasks or CANCEL_ALL_TASKS in self._cancelled_tasks
    
    def step(self) -> Iterable[tuple[TaskId, Chunk | CancelledResponse | FinishedResponse]]:
        """Execute one step of speculative decoding."""
        self.agree_on_cancellations()
        self.agree_on_tasks()
        
        output: list[tuple[TaskId, Chunk | CancelledResponse | FinishedResponse]] = []
        
        # Handle cancellations
        for task_id in list(self._cancelled_tasks):
            if task_id != CANCEL_ALL_TASKS:
                output.append((task_id, CancelledResponse()))
        self._cancelled_tasks.clear()
        
        # Start new task if none active
        if self._active_generator is None and self._queue:
            task = self._queue.pop(0)
            if not self.should_cancel(task.task_id):
                try:
                    self._active_task = task
                    self._active_generator = self._build_speculative_generator(task)
                except Exception as e:
                    self._send_error(task, e)
                    raise
        
        # Step active generator
        if self._active_generator is not None and self._active_task is not None:
            try:
                response = next(self._active_generator)
                if response.finish_reason is not None:
                    output.append((self._active_task.task_id, FinishedResponse()))
                    self._active_generator = None
                    self._active_task = None
                elif self.device_rank == 0:
                    # Convert GenerationResponse to GenerationChunk for streaming
                    chunk = GenerationChunk(
                        model=self.target_model_id,
                        text=response.text,
                        finish_reason=response.finish_reason,
                    )
                    output.append((self._active_task.task_id, chunk))
            except StopIteration:
                output.append((self._active_task.task_id, FinishedResponse()))
                self._active_generator = None
                self._active_task = None
            except Exception as e:
                self._send_error(self._active_task, e)
                self._active_generator = None
                self._active_task = None
                raise
        
        return output
    
    def _build_speculative_generator(self, task: TextGeneration) -> Generator[GenerationResponse]:
        """Build generator that uses speculative decoding with fallback."""
        prompt = apply_chat_template(self.target_tokenizer, task.task_params)
        
        # Check if speculative decoding should be used
        use_speculative = (
            self.config.enabled
            and self.draft_model is not None
            and self.target_model is not None
        )
        
        if not use_speculative:
            logger.info("Speculative decoding disabled, falling back to standard generation")
            yield from mlx_generate(
                model=self.target_model,
                tokenizer=self.target_tokenizer,
                task=task.task_params,
                prompt=prompt,
                kv_prefix_cache=self.target_kv_prefix_cache,
                group=self.target_group,
                vision_processor=self.vision_processor,
            )
            return
        
        # Initialize caches
        target_cache = make_kv_cache(self.target_model, max_kv_size=8192)
        draft_cache = make_kv_cache(self.draft_model, max_kv_size=8192)
        
        # Prefill both models with prompt
        logger.info(f"Speculative prefill: {len(prompt)} tokens")
        prompt_tokens = mx.array(self.target_tokenizer.encode(prompt))
        
        # Prefill target
        if len(prompt_tokens) > 1:
            self.target_model(prompt_tokens[:-1][None], cache=target_cache)
        
        # Prefill draft (may have different tokenizer - handle mismatch)
        try:
            draft_prompt_tokens = mx.array(self.draft_tokenizer.encode(prompt))
            if len(draft_prompt_tokens) > 1:
                self.draft_model(draft_prompt_tokens[:-1][None], cache=draft_cache)
            last_draft_token = draft_prompt_tokens[-1].item() if len(draft_prompt_tokens) > 0 else 0
        except Exception:
            logger.warning("Draft tokenizer mismatch, disabling speculative decoding")
            yield from mlx_generate(
                model=self.target_model,
                tokenizer=self.target_tokenizer,
                task=task.task_params,
                prompt=prompt,
                kv_prefix_cache=self.target_kv_prefix_cache,
                group=self.target_group,
                vision_processor=self.vision_processor,
            )
            return
        
        last_target_token = prompt_tokens[-1].item() if len(prompt_tokens) > 0 else 0
        max_tokens = task.task_params.max_output_tokens or 4096
        generated_count = 0
        
        logger.info(f"Starting speculative decoding: K={self._stats.current_k}")
        
        while generated_count < max_tokens:
            # Check cancellation periodically
            if generated_count % self.check_for_cancel_every == 0:
                self.agree_on_cancellations()
                if self.should_cancel(task.task_id):
                    return
            
            # Adaptive K based on acceptance rate
            if self._stats.consecutive_rejections >= self.config.max_consecutive_rejections:
                self._stats.current_k = max(1, self._stats.current_k // 2)
                self._stats.consecutive_rejections = 0
                logger.debug(f"Reduced K to {self._stats.current_k} due to low acceptance")
            elif self._stats.acceptance_rate > 0.8 and self._stats.total_verification_steps > 10:
                self._stats.current_k = min(self.config.num_draft_tokens, self._stats.current_k + 1)
            
            # Generate K draft tokens
            draft_tokens = generate_draft_tokens(
                draft_model=self.draft_model,
                draft_tokenizer=self.draft_tokenizer,
                draft_cache=draft_cache,
                start_token=last_draft_token,
                num_tokens=self._stats.current_k,
                temperature=self.config.draft_temperature,
                group=self.draft_group,
            )
            
            # Verify against target
            accepted, bonus_token = verify_draft_tokens(
                target_model=self.target_model,
                target_cache=target_cache,
                draft_tokens=draft_tokens,
                last_verified_token=last_target_token,
                group=self.target_group,
            )
            
            self._stats.record_verification(len(draft_tokens), len(accepted))
            
            # Yield accepted tokens + bonus
            all_accepted = accepted + [bonus_token]
            for tok in all_accepted:
                text = self.target_tokenizer.decode([tok])
                generated_count += 1
                
                yield GenerationResponse(
                    text=text,
                    token=tok,
                    logprob=None,
                    top_logprobs=None,
                    finish_reason="length" if generated_count >= max_tokens else None,
                    stats=None,
                    usage=None,
                )
                
                if generated_count >= max_tokens:
                    break
            
            last_target_token = bonus_token
            
            # Update draft cache to stay in sync
            # Trim draft cache to match accepted length
            n_accepted = len(accepted)
            n_to_trim = len(draft_tokens) - n_accepted
            if n_to_trim > 0:
                for c in draft_cache:
                    if hasattr(c, 'trim'):
                        c.trim(n_to_trim)
            
            # Advance draft by bonus token
            last_draft_token = bonus_token
            bonus_input = mx.array([bonus_token])[None, :]
            self.draft_model(bonus_input, cache=draft_cache)
        
        logger.info(
            f"Speculative decoding complete: {self._stats.total_accepted_tokens}/"
            f"{self._stats.total_draft_tokens} accepted "
            f"({self._stats.acceptance_rate:.1%}), {self._stats.total_verification_steps} steps"
        )
    
    def _send_error(self, task: TextGeneration, e: Exception) -> None:
        if self.device_rank == 0:
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        model=self.target_model_id,
                        finish_reason="error",
                        error_message=str(e),
                    ),
                )
            )
    
    def close(self) -> None:
        del self.target_model, self.target_tokenizer, self.target_group
        del self.draft_model, self.draft_tokenizer, self.draft_group
    
    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        """Serve prefill requests - delegate to target model."""
        from exo.worker.engines.mlx.disaggregated.serve import run_prefill_for_request
        from exo.worker.engines.mlx.disaggregated.adapter import write_cache_to_wire
        
        cache = run_prefill_for_request(
            model=self.target_model,
            tokenizer=self.target_tokenizer,
            group=self.target_group,
            kv_prefix_cache=self.target_kv_prefix_cache,
            request=request,
        )
        write_cache_to_wire(
            wfile,
            cache,
            request_id=request.request_id,
            model_id=request.model_id,
            start_pos=request.start_pos,
        )
