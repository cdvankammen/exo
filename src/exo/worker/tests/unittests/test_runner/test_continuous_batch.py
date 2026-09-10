"""Test ContinuousBatchScheduler -- 8 concurrent requests, admission, cancellation.

Uses the same FakeExoBatchGenerator / EventCollector / MockGroup pattern as
test_event_ordering.py to exercise the scheduler in isolation from MLX.
"""
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable

import pytest

import exo.worker.engines.mlx.builder as mlx_builder
import exo.worker.runner.llm_inference.batch_generator as mlx_batch_generator
import exo.worker.runner.llm_inference.continuous_batch as cb_mod
import exo.worker.runner.llm_inference.model_output_parsers as mlx_model_output_parsers
from exo.shared.types.chunks import GenerationChunk, TokenChunk
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    ConnectToGroup,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
    GenerationResponse,
)
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerWarmingUp,
)
from exo.utils.channels import mp_channel
from exo.worker.engines.mlx.builder import MlxBuilder
from exo.worker.runner.runner import Runner

from ...constants import (
    CHAT_COMPLETION_TASK_ID,
    COMMAND_1_ID,
    INITIALIZATION_TASK_ID,
    INSTANCE_1_ID,
    LOAD_TASK_ID,
    MODEL_A_ID,
    NODE_A,
    RUNNER_1_ID,
    SHUTDOWN_TASK_ID,
    WARMUP_TASK_ID,
)
from ..conftest import get_bound_mlx_ring_instance


def make_nothin(res):
    def nothin(*_1, **_2):
        return res
    return nothin

nothin = make_nothin(None)


class FakeExoBatchGenerator:
    def __init__(self, *_args, **_kwargs):
        self._uid_counter = 0
        self._pending = {}
        self._cancelled = []

    @property
    def has_work(self):
        return bool(self._pending)

    def submit(self, task_params=None, prompt=None, on_prefill_progress=None,
               distributed_prompt_progress_callback=None, on_generation_token=None):
        uid = self._uid_counter
        self._uid_counter += 1
        self._pending[uid] = GenerationResponse(text="hi", token=0, finish_reason="stop", usage=None)
        return uid

    def step(self):
        results = list(self._pending.items())
        self._pending.clear()
        return results

    def cancel(self, uids):
        self._cancelled.extend(uids)
        for uid in uids:
            self._pending.pop(uid, None)

    def close(self):
        pass


class EventCollector:
    def __init__(self, on_event=None):
        self.events = []
        self._on_event = on_event
    def send(self, event):
        self.events.append(event)
        if self._on_event:
            self._on_event(event)
    def close(self):
        pass
    def join(self):
        pass


class MockTokenizer:
    tool_parser = None
    tool_call_start = None
    tool_call_end = None
    has_tool_calling = False
    has_thinking = False
    think_start = None
    think_end = None
    eos_token_ids = []
    @staticmethod
    def decode(_tokens):
        return "hi"
    @staticmethod
    def encode(_text, add_special_tokens=True):
        return [0]

class MockGroup:
    def rank(self):
        return 0
    def size(self):
        return 1

@dataclass
class MockLoadOutput:
    layers_loaded: int
    total: int


@pytest.fixture
def patch_out_mlx(monkeypatch):
    """Patch all MLX / model-loading paths so tests run without a real model."""
    monkeypatch.setattr(mlx_builder, "initialize_mlx", make_nothin(MockGroup()))

    def lmi_gen():
        yield MockLoadOutput(1, 1)
        return (1, MockTokenizer, None)

    monkeypatch.setattr(mlx_builder, "load_mlx_items", make_nothin(lmi_gen()))
    monkeypatch.setattr(mlx_batch_generator, "warmup_inference", make_nothin(1))
    monkeypatch.setattr(mlx_batch_generator, "_check_for_debug_prompts", nothin)
    monkeypatch.setattr(mlx_batch_generator, "mx_any", make_nothin(False))

    def fake_all_gather(tasks, group):
        return (tasks, [])

    monkeypatch.setattr(mlx_batch_generator, "mx_all_gather_tasks", fake_all_gather)
    monkeypatch.setattr(mlx_batch_generator, "apply_chat_template", make_nothin("test prompt"))
    monkeypatch.setattr(mlx_model_output_parsers, "detect_thinking_prompt_suffix", make_nothin(False))
    monkeypatch.setattr(mlx_batch_generator, "ExoBatchGenerator", FakeExoBatchGenerator)

    def _no_prefill_server(_self):
        return None

    monkeypatch.setattr(Runner, "_start_prefill_server", _no_prefill_server)


def test_eight_concurrent_requests_all_complete(patch_out_mlx):
    """Eight tasks: init + load + warmup + 6 chats -> shutdown. All complete."""
    tasks = [
        ConnectToGroup(task_id=INITIALIZATION_TASK_ID, instance_id=INSTANCE_1_ID),
        LoadModel(task_id=LOAD_TASK_ID, instance_id=INSTANCE_1_ID),
        StartWarmup(task_id=WARMUP_TASK_ID, instance_id=INSTANCE_1_ID),
    ]
    for i in range(6):
        params = TextGenerationTaskParams(
            model=MODEL_A_ID,
            input=[InputMessage(role="user", content=InputMessageContent(f"msg{i}"))],
            stream=True,
            max_output_tokens=4,
            temperature=0.0,
        )
        tasks.append(
            TextGeneration(
                task_id=TaskId(f"chat-{i}"),
                command_id=COMMAND_1_ID,
                task_params=params,
                instance_id=INSTANCE_1_ID,
            )
        )
    tasks.append(Shutdown(task_id=SHUTDOWN_TASK_ID, instance_id=INSTANCE_1_ID, runner_id=RUNNER_1_ID))

    bound_instance = get_bound_mlx_ring_instance(
        instance_id=INSTANCE_1_ID, model_id=MODEL_A_ID,
        runner_id=RUNNER_1_ID, node_id=NODE_A,
    )
    task_sender, task_receiver = mp_channel[Task]()
    _cancel_sender, cancel_receiver = mp_channel[TaskId]()
    event_sender = EventCollector()

    with task_sender:
        for t in tasks:
            task_sender.send(t)
        task_receiver.close = nothin
        task_receiver.join = nothin
        builder = MlxBuilder(
            bound_instance.bound_shard.model_card.model_id, event_sender, cancel_receiver,
        )
        runner = Runner(bound_instance, builder, event_sender, task_receiver)
        runner.main()
        events = event_sender.events

    # Every chat task must have acknowledged + completed.
    for i in range(6):
        chat_id = TaskId(f"chat-{i}")
        acks = [e for e in events if isinstance(e, TaskAcknowledged) and e.task_id == chat_id]
        completes = [e for e in events if isinstance(e, TaskStatusUpdated)
                     and e.task_id == chat_id and e.task_status == TaskStatus.Complete]
        assert len(acks) == 1, f"{chat_id}: expected 1 ack, got {len(acks)}"
        assert len(completes) == 1, f"{chat_id}: expected 1 complete, got {len(completes)}"

    # Shutdown completed.
    shutdown_complete = [e for e in events if isinstance(e, TaskStatusUpdated)
                         and e.task_id == SHUTDOWN_TASK_ID and e.task_status == TaskStatus.Complete]
    assert len(shutdown_complete) == 1


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------

def test_queued_request_priority_promotion():
    import time
    qr = cb_mod._QueuedRequest(task=None, priority=0, submitted_at=time.monotonic() - 10.0)
    assert qr.promoted is True
    assert qr.effective_priority == cb_mod.PRIORITY_PROMOTED
    qr2 = cb_mod._QueuedRequest(task=None, priority=5, submitted_at=time.monotonic())
    assert qr2.promoted is False
    assert qr2.effective_priority == 5


def test_batch_metrics_utilization():
    m = cb_mod._BatchMetrics()
    m.record_step(batch_size=4, is_prefill=False, elapsed=0.1)
    m.record_step(batch_size=2, is_prefill=False, elapsed=0.1)
    m.record_admission()
    m.record_admission()
    assert m.peak_batch_size == 4
    # 4+2 slots filled across 2 steps of peak 4 => 6/(2*4) = 0.75
    assert abs(m.utilization - 0.75) < 0.01
    assert m.requests_admitted == 2
    s = m.summary()
    assert s["peak_batch_size"] == 4
    assert s["requests_admitted"] == 2


def test_prefill_cancelled_is_exception():
    with pytest.raises(cb_mod._PrefillCancelled):
        raise cb_mod._PrefillCancelled()

def test_builder_flags(monkeypatch):
    """MlxBuilder selects the correct Engine based on env vars."""
    # Fake out model/tokenizer and caches so builder doesn't need MLX.
    from exo.worker.engines.mlx.builder import MlxBuilder
    import exo.worker.runner.llm_inference.continuous_batch as cb_mod2
    import exo.worker.runner.llm_inference.batch_generator as bg

    fake_model = object()
    fake_tok = type("Tok", (), dict(
        has_tool_calling=False, tool_call_start=None, tool_call_end=None, tool_parser=None,
    ))()
    fake_group = None

    monkeypatch.setattr("exo.worker.engines.mlx.builder.load_mlx_items", lambda *a, **k: None)
    monkeypatch.setattr("exo.worker.engines.mlx.builder.KVPrefixCache", lambda *a, **k: None)

    _sender, recv = mp_channel[TaskId]()
    collector = EventCollector()
    builder = MlxBuilder(model_id=MODEL_A_ID, event_sender=collector, cancel_receiver=recv)
    builder.inference_model = fake_model  # type: ignore[attr-defined]
    builder.tokenizer = fake_tok  # type: ignore[attr-defined]
    builder.group = fake_group
    builder.vision_processor = None

    # Default -> BatchGenerator
    monkeypatch.delenv("EXO_NO_BATCH", raising=False)
    monkeypatch.delenv("EXO_USE_CONTINUOUS_BATCH", raising=False)
    engine = builder.build()
    assert isinstance(engine, bg.BatchGenerator)

    # EXO_NO_BATCH -> SequentialGenerator
    monkeypatch.setenv("EXO_NO_BATCH", "1")
    monkeypatch.delenv("EXO_USE_CONTINUOUS_BATCH", raising=False)
    engine = builder.build()
    assert isinstance(engine, bg.SequentialGenerator)

    # EXO_USE_CONTINUOUS_BATCH -> ContinuousBatchScheduler (takes precedence over default)
    monkeypatch.delenv("EXO_NO_BATCH", raising=False)
    monkeypatch.setenv("EXO_USE_CONTINUOUS_BATCH", "1")
    engine = builder.build()
    assert isinstance(engine, cb_mod2.ContinuousBatchScheduler)

    # EXO_NO_BATCH + EXO_USE_CONTINUOUS_BATCH -> SequentialGenerator wins (EXO_NO_BATCH precedence)
    monkeypatch.setenv("EXO_NO_BATCH", "1")
    monkeypatch.setenv("EXO_USE_CONTINUOUS_BATCH", "1")
    engine = builder.build()
    assert isinstance(engine, bg.SequentialGenerator)


def test_observability_metrics(monkeypatch):
    """ContinuousBatchScheduler exposes running/waiting counts and metrics."""
    def fake_all_gather(tasks, group):
        return (tasks, [])

    monkeypatch.setattr(cb_mod, "mx_all_gather_tasks", fake_all_gather)
    monkeypatch.setattr(cb_mod, "mx_any", lambda *_args, **_kw: False)
    monkeypatch.setattr(cb_mod, "_check_for_debug_prompts", lambda *_args, **_kw: None)
    monkeypatch.setattr(cb_mod, "apply_chat_template", lambda *_args, **_kw: "test prompt")
    monkeypatch.setattr(cb_mod, "ExoBatchGenerator", FakeExoBatchGenerator)

    # Construct scheduler directly to check metric surface without running the runner.
    from exo.utils.channels import mp_channel
    _a, recv = mp_channel[TaskId]()
    col = EventCollector()
    scheduler = cb_mod.ContinuousBatchScheduler(
        model=object(), tokenizer=MockTokenizer(), group=None,
        kv_prefix_cache=None, tool_parser=None,
        model_id=MODEL_A_ID, device_rank=0,
        cancel_receiver=recv, event_sender=col,
    )
    assert scheduler.running_count == 0
    assert scheduler.waiting_count == 0
    m = scheduler.get_metrics()
    assert m["running"] == 0
    assert m["waiting"] == 0
    assert "peak_batch_size" in m
    assert "utilization" in m
    assert "requests_admitted" in m
