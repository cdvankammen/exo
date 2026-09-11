"""Continuous Batching Scheduler for LLM Inference (vLLM-style).

Phase 2.1 of the exo inference engine refactoring. Replaces SequentialGenerator
with a scheduler that processes N requests simultaneously in one forward pass,
then immediately admits new requests as slots free up -- never waiting for the
full batch to drain.  Provides 2-4x throughput improvement under concurrent load.

Key design (vLLM port):
- running dict: active batch (TaskId -> _RunningRequest)
- waiting deque: FIFO queue with starvation-prevention priority boosting
- step(): one forward pass over all running requests, then dynamic admission
- Batch metrics for observability (utilization, prefill/decode timing)

Usage:
    Export EXO_USE_CONTINUOUS_BATCH=1 to select this scheduler instead of
    BatchGenerator in MlxBuilder.build(). EXO_NO_BATCH=1 still takes
    precedence and forces SequentialGenerator.
"""

import itertools
import time
from collections import deque
from dataclasses import dataclass, field
from typing import BinaryIO, Iterator

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.constants import max_concurrent_requests
from exo.shared.types.chunks import ErrorChunk, GenerationChunk, PrefillProgressChunk
from exo.shared.types.common import ModelId
from exo.shared.types.events import ChunkGenerated, Event
from exo.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    GenerationTask,
    TaskId,
    TextGeneration,
)
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
    GenerationResponse,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.base import Engine
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.disaggregated.adapter import write_cache_to_wire
from exo.worker.engines.mlx.disaggregated.serve import run_prefill_for_request
from exo.worker.engines.mlx.generator.batch_generate import ExoBatchGenerator
from exo.worker.engines.mlx.generator.generate import warmup_inference
from exo.worker.engines.mlx.types import Model
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    mx_all_gather_tasks,
    mx_any,
)
from exo.worker.engines.mlx.vision import VisionProcessor
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.batch_generator import (
    GeneratorQueue,
    _check_for_debug_prompts,
)
from exo.worker.runner.llm_inference.model_output_parsers import (
    apply_all_parsers,
    map_responses_to_chunks,
)
from exo.worker.runner.llm_inference.tool_parsers import ToolParser


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRIORITY_PROMOTION_THRESHOLD: float = 5.0
"""Seconds a request must wait before its effective priority is boosted to
prevent starvation when the batch is fully occupied by faster requests."""

PRIORITY_PROMOTED: int = -1
"""Effective priority value for promoted (starved) requests."""

# Type alias for step() output items -- plain assignment for Python < 3.12 compat.
_BatchOutput = tuple["TaskId", "GenerationChunk | CancelledResponse | FinishedResponse"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _PrefillCancelled(Exception):
    """Raised when a prefill is cancelled before it starts generation."""


@dataclass
class _QueuedRequest:
    """A request waiting for a batch slot."""

    task: TextGeneration
    submitted_at: float = field(default_factory=time.monotonic)
    priority: int = 0

    @property
    def wait_time(self) -> float:
        return time.monotonic() - self.submitted_at

    @property
    def promoted(self) -> bool:
        return self.wait_time > PRIORITY_PROMOTION_THRESHOLD

    @property
    def effective_priority(self) -> int:
        if self.promoted:
            return min(self.priority, PRIORITY_PROMOTED)
        return self.priority


@dataclass
class _RunningRequest:
    """An active request in the continuous batch."""

    task: TextGeneration
    uid: int
    started_at: float = field(default_factory=time.monotonic)
    queue: GeneratorQueue[GenerationResponse] = field(
        default_factory=lambda: GeneratorQueue()
    )
    output_generator: Iterator[GenerationChunk | None] | None = None
    tokens_generated: int = 0
    is_prefill_pending: bool = True

    def init_output_generator(
        self,
        output_generator: Iterator[GenerationChunk | None],
    ) -> None:
        self.output_generator = output_generator

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at


@dataclass
class _BatchMetrics:
    """Observability counters for the continuous batch scheduler.

    Tracks utilization, prefill/decode timing, and admission stats
    for monitoring batch efficiency.
    """

    total_prefill_time: float = 0.0
    total_decode_time: float = 0.0
    requests_admitted: int = 0
    peak_batch_size: int = 0
    _step_count: int = 0
    _total_slot_filled: int = 0

    def record_step(
        self, batch_size: int, is_prefill: bool, elapsed: float
    ) -> None:
        self._step_count += 1
        self._total_slot_filled += batch_size
        if batch_size > self.peak_batch_size:
            self.peak_batch_size = batch_size
        if is_prefill:
            self.total_prefill_time += elapsed
        else:
            self.total_decode_time += elapsed

    def record_admission(self) -> None:
        self.requests_admitted += 1

    @property
    def utilization(self) -> float:
        if self._step_count == 0 or self.peak_batch_size == 0:
            return 0.0
        return self._total_slot_filled / (self._step_count * self.peak_batch_size)

    def summary(self) -> dict:
        return {
            "requests_admitted": self.requests_admitted,
            "peak_batch_size": self.peak_batch_size,
            "utilization": round(self.utilization, 3),
            "total_prefill_time": round(self.total_prefill_time, 4),
            "total_decode_time": round(self.total_decode_time, 4),
        }


# ---------------------------------------------------------------------------
# Main scheduler
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class ContinuousBatchScheduler(Engine):
    """vLLM-inspired continuous batching scheduler.

    Processes ``_max_batch_size`` requests simultaneously via
    :class:`ExoBatchGenerator`, then immediately admits new waiting requests
    as slots free up -- never waiting for the full batch to drain.

    Inherits the ``Engine`` interface so it can be used as a drop-in
    replacement for ``SequentialGenerator`` / ``BatchGenerator`` in the
    runner's main loop.

    Scheduling policy:
        1. On each ``step()`` call, drain all available results from the
           engine (``_gen.step()``) and emit ``FinishedResponse`` for any
           completed requests.
        2. Remove finished and cancelled requests from the running batch.
        3. Admit new requests from the waiting queue up to the batch size
           limit, FIFO order with starvation protection.
        4. Emit cancellations for any tasks that arrived while the batch
           was full.
    """

    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None
    tool_parser: ToolParser | None
    model_id: ModelId
    device_rank: int
    cancel_receiver: MpReceiver[TaskId]
    event_sender: MpSender[Event]
    vision_processor: VisionProcessor | None = None
    check_for_cancel_every: int = 50
    _max_batch_size: int = field(default_factory=max_concurrent_requests, init=True)

    # -- internal state (init=False) --
    _gen: ExoBatchGenerator = field(init=False)
    _running: dict[TaskId, _RunningRequest] = field(default_factory=dict, init=False)
    _waiting: deque[_QueuedRequest] = field(default_factory=deque, init=False)
    _queue: deque[TextGeneration] = field(default_factory=deque, init=False)
    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _all_tasks: dict[TaskId, TextGeneration] = field(default_factory=dict, init=False)
    _maybe_queue: list[TextGeneration] = field(default_factory=list, init=False)
    _maybe_cancel: list[TextGeneration] = field(default_factory=list, init=False)
    _metrics: _BatchMetrics = field(default_factory=_BatchMetrics, init=False)
    _uid_to_task_id: dict[int, TaskId] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._gen = ExoBatchGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            kv_prefix_cache=self.kv_prefix_cache,
            vision_processor=self.vision_processor,
        )

    # ------------------------------------------------------------------
    # Engine interface
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        self.check_for_cancel_every = warmup_inference(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            model_id=self.model_id,
        )

    def submit(self, task: GenerationTask) -> None:
        assert isinstance(task, TextGeneration)
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self._all_tasks[task.task_id] = task
        self._maybe_queue.append(task)

    def agree_on_tasks(self) -> None:
        """Agree between all ranks about the task ordering (some may have
        received in different order or not at all)."""
        agreed, different = mx_all_gather_tasks(self._maybe_queue, self.group)
        self._queue.extend(agreed)
        self._maybe_queue = list(different)

    def agree_on_cancellations(self) -> None:
        """Agree between all ranks about which tasks to cancel."""
        has_cancel_all = False
        for task_id in self.cancel_receiver.collect():
            if task_id == CANCEL_ALL_TASKS:
                has_cancel_all = True
                continue
            if task_id in self._all_tasks:
                self._maybe_cancel.append(self._all_tasks[task_id])

        if mx_any(has_cancel_all, self.group):
            self._cancelled_tasks.add(CANCEL_ALL_TASKS)

        agreed, different = mx_all_gather_tasks(self._maybe_cancel, self.group)
        self._cancelled_tasks.update(task.task_id for task in agreed)
        self._maybe_cancel = list(different)

    def step(
        self,
    ) -> Iterator[_BatchOutput]:
        """Process one forward pass and admit new requests.

        1.  Run ``_gen.step()`` to advance all running requests by one
            forward pass and collect results.
        2.  Emit ``FinishedResponse`` / ``CancelledResponse`` for
            completed / cancelled requests.
        3.  Immediately admit waiting requests into freed slots.
        4.  Return all output chunks for the runner to forward to clients.

        This is the core of the continuous batching loop: new requests
        are injected mid-generation as soon as capacity is available.
        """
        output: list[_BatchOutput] = []

        # Agree with the other ranks on newly-submitted tasks, then stage
        # them in the waiting queue for admission as batch slots free up.
        # (Same agreement cadence as BatchGenerator.step: agree whenever the
        # agreed queue is drained, so all ranks stay in lockstep.)
        if not self._queue:
            self.agree_on_tasks()
        while self._queue:
            self._waiting.append(_QueuedRequest(task=self._queue.popleft()))

        batch_size = len(self._running)
        is_prefill_pending = any(r.is_prefill_pending for r in self._running.values())

        # -- advance the engine one step --
        if self._gen.has_work:
            _step_tic = time.perf_counter()
            results = self._gen.step()
            _step_elapsed = time.perf_counter() - _step_tic
            self._metrics.record_step(batch_size, is_prefill_pending, _step_elapsed)

            for uid, response in results:
                task_id = self._uid_to_task_id.get(uid)
                if task_id is None or task_id not in self._running:
                    logger.warning(
                        f"ContinuousBatchScheduler: uid={uid} (task_id={task_id}) "
                        f"not found in active requests"
                    )
                    continue

                req = self._running[task_id]
                req.queue.push(response)

                # Drain the output parser generator for this request.
                if req.output_generator is not None:
                    while (
                        parsed := next(req.output_generator, None)
                    ) is not None:
                        output.append((task_id, parsed))

                req.is_prefill_pending = False
                req.tokens_generated += 1

                if response.finish_reason is not None:
                    output.append((task_id, FinishedResponse()))
                    self._cleanup_running(task_id)

        # -- admit new requests from the waiting queue --
        self._admit_from_waiting(output)

        # -- apply pending cancellations --
        cancel_output = list(self._apply_cancellations())

        return filter(
            lambda chunk: (
                not isinstance(chunk[1], GenerationChunk) or self.device_rank == 0
            ),
            itertools.chain(output, cancel_output),
        )

    def close(self) -> None:
        self._gen.close()
        del self.model, self.tokenizer, self.group

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        cache = run_prefill_for_request(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            kv_prefix_cache=self.kv_prefix_cache,
            request=request,
        )
        write_cache_to_wire(
            wfile,
            cache,
            request_id=request.request_id,
            model_id=request.model_id,
            start_pos=request.start_pos,
        )

    # ------------------------------------------------------------------
    # Scheduling helpers
    # ------------------------------------------------------------------

    def _admit_from_waiting(self, output: list[_BatchOutput]) -> None:
        """Pull from the waiting queue and start running requests until
        the batch is full or the queue is empty."""
        while self._waiting and len(self._running) < self._max_batch_size:
            queued = self._waiting[0]
            task_id = queued.task.task_id

            # Skip already-cancelled tasks without starting them.
            if task_id in self._cancelled_tasks:
                self._waiting.popleft()
                output.append((task_id, CancelledResponse()))
                continue

            self._waiting.popleft()
            try:
                self._admit_request(queued)
                self._metrics.record_admission()
            except _PrefillCancelled:
                logger.debug(
                    f"ContinuousBatchScheduler: prefill cancelled for {task_id}"
                )
                continue
            except Exception as e:
                self._send_error(queued.task, e)
                raise

    def _admit_request(self, queued: _QueuedRequest) -> None:
        """Start a queued request in the batch engine."""
        task = queued.task
        uid = self._start_task(task)

        req = _RunningRequest(
            task=task,
            uid=uid,
            started_at=queued.submitted_at,
        )
        queue = req.queue
        if task.task_params.bench:
            output_generator: Iterator[GenerationChunk | None] = map(
                lambda r: map_responses_to_chunks(r, self.model_id), queue.gen()
            )
        else:
            output_generator = apply_all_parsers(
                queue.gen(),
                apply_chat_template(self.tokenizer, task.task_params),
                self.tool_parser,
                self.tokenizer,
                type(self.model),
                self.model_id,
                task.task_params.tools,
            )
        req.init_output_generator(output_generator)

        self._running[task.task_id] = req
        self._uid_to_task_id[uid] = task.task_id

    def _start_task(self, task: TextGeneration) -> int:
        """Submit a task to the underlying ExoBatchGenerator."""
        _check_for_debug_prompts(task.task_params)
        prompt = apply_chat_template(self.tokenizer, task.task_params)

        def on_prefill_progress(processed: int, total: int) -> None:
            if self.device_rank == 0:
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=task.command_id,
                        chunk=PrefillProgressChunk(
                            model=self.model_id,
                            processed_tokens=processed,
                            total_tokens=total,
                        ),
                    )
                )

        def distributed_prompt_progress_callback() -> None:
            self.agree_on_cancellations()
            if self.should_cancel(task.task_id):
                raise _PrefillCancelled()
            self.agree_on_tasks()

        tokens_since_cancel_check = self.check_for_cancel_every

        def on_generation_token() -> None:
            nonlocal tokens_since_cancel_check
            tokens_since_cancel_check += 1
            if tokens_since_cancel_check >= self.check_for_cancel_every:
                tokens_since_cancel_check = 0
                self.agree_on_cancellations()
                if self.should_cancel(task.task_id):
                    self._cancelled_tasks.add(task.task_id)
                self.agree_on_tasks()

        return self._gen.submit(
            task_params=task.task_params,
            prompt=prompt,
            on_prefill_progress=on_prefill_progress,
            distributed_prompt_progress_callback=distributed_prompt_progress_callback,
            on_generation_token=on_generation_token,
        )

    def _cleanup_running(self, task_id: TaskId) -> None:
        req = self._running.pop(task_id, None)
        if req is not None:
            self._uid_to_task_id.pop(req.uid, None)

    def _apply_cancellations(self) -> Iterator[_BatchOutput]:
        """Remove cancelled tasks from both running and waiting sets."""
        if not self._cancelled_tasks:
            return iter([])

        cancel_all = CANCEL_ALL_TASKS in self._cancelled_tasks

        results: list[_BatchOutput] = []
        uids_to_cancel: list[int] = []

        # Cancel running requests.
        for task_id, req in list(self._running.items()):
            if task_id in self._cancelled_tasks or cancel_all:
                uids_to_cancel.append(req.uid)
                results.append((task_id, CancelledResponse()))
                self._cleanup_running(task_id)

        if uids_to_cancel:
            self._gen.cancel(uids_to_cancel)

        # Purge waiting queue of cancelled requests.
        self._waiting = deque(
            q
            for q in self._waiting
            if q.task.task_id not in self._cancelled_tasks
            and not cancel_all
        )

        # Report any task IDs that were neither running nor waiting.
        already_cancelled = {tid for tid, _ in results}
        for tid in self._cancelled_tasks:
            if tid != CANCEL_ALL_TASKS and tid not in already_cancelled:
                results.append((tid, CancelledResponse()))

        self._cancelled_tasks.clear()
        return iter(results)

    def _send_error(self, task: TextGeneration, e: Exception) -> None:
        if self.device_rank == 0:
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        model=self.model_id,
                        finish_reason="error",
                        error_message=str(e),
                    ),
                )
            )

    # ------------------------------------------------------------------
    # Observability properties
    # ------------------------------------------------------------------

    @property
    def batch_utilization(self) -> float:
        """Fraction of batch slots that were filled on average."""
        return self._metrics.utilization

    @property
    def waiting_count(self) -> int:
        """Number of requests waiting for a batch slot."""
        return len(self._waiting)

    @property
    def running_count(self) -> int:
        """Number of requests actively in the batch."""
        return len(self._running)

    def get_metrics(self) -> dict:
        """Return a snapshot of batch scheduler metrics."""
        m = self._metrics.summary()
        m["running"] = self.running_count
        m["waiting"] = self.waiting_count
        return m
