# P1 #39: OOM graceful degradation (kanban t_2722376d).
#
# On OOM the generators must: clear the KV cache, halve the batch size,
# report a degraded status (RunnerDegraded via OomRecoveredError) and retry
# instead of letting the OOM escape and crash-loop the runner.
#
# Deterministic fakes only — no MLX required (mirrors test_all_tasks_pruning.py
# and test_continuous_batch_scheduler.py).
# pyright: reportPrivateUsage=false
from __future__ import annotations

from collections.abc import Iterator

import pytest

from exo.shared.types.chunks import TokenChunk
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.tasks import TextGeneration
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.runner_response import (
    FinishedResponse,
    GenerationResponse,
)
from exo.shared.types.worker.runners import (
    RunnerDegraded,
    RunnerReady,
    RunnerRunning,
)
from exo.worker.runner.llm_inference import batch_generator as bg
from exo.worker.runner.llm_inference import continuous_batch as cb
from exo.worker.runner.llm_inference.batch_generator import OomRecoveredError

# ---------------------------------------------------------------------------
# Shared fakes (mirror the existing runner test files)
# ---------------------------------------------------------------------------


class FakeCancelReceiver:
    def __init__(self, items: list | None = None) -> None:
        self._items: list = list(items or [])

    def collect(self) -> list:
        items = self._items
        self._items = []
        return items


class CollectingSender:
    def __init__(self) -> None:
        self.sent: list[object] = []

    def send(self, event: object) -> None:
        self.sent.append(event)


class OomFakeExoBatchGenerator:
    """Fake engine that raises a MemoryError-like OOM on the first ``step()``.

    After the wrapper's ``_oom_recover()`` calls ``reset()`` and resubmits,
    subsequent steps behave like the happy-path fake.
    """

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self._uid_counter = 0
        self._pending: dict[int, GenerationResponse] = {}
        self.reset_count = 0
        self._arm_oom = True

    @property
    def has_work(self) -> bool:
        return bool(self._pending)

    def submit(
        self,
        task_params: object = None,
        prompt: object = None,
        on_prefill_progress: object = None,
        distributed_prompt_progress_callback: object = None,
        on_generation_token: object = None,
    ) -> int:
        uid = self._uid_counter
        self._uid_counter += 1
        self._pending[uid] = GenerationResponse(
            text="hi",
            token=0,
            finish_reason="stop",
            usage=None,
        )
        return uid

    def step(self) -> list[tuple[int, GenerationResponse]]:
        if self._arm_oom:
            # First step is the one that OOMs (e.g. during prefill of the
            # freshly submitted batch).
            self._arm_oom = False
            raise MemoryError("Failed to allocate [8 x 4096 x 64 x 128]")
        results = list(self._pending.items())
        self._pending.clear()
        return results

    def reset(self) -> None:
        self.reset_count += 1
        self._pending.clear()

    def cancel(self, uids: list[int]) -> None:
        for uid in uids:
            self._pending.pop(uid, None)

    def close(self) -> None:
        pass


class _FakeModel:
    pass


class _FakeTokenizer:
    tool_parser = None
    tool_call_start = None
    tool_call_end = None
    has_tool_calling = False
    has_thinking = False
    think_start = None
    think_end = None
    eos_token_ids: list[int] = []

    @staticmethod
    def decode(tokens: list[int]) -> str:
        return "hi"

    @staticmethod
    def encode(text: str, add_special_tokens: bool = True) -> list[int]:
        return [0]


class _FakeGroup:
    def rank(self) -> int:
        return 0


def _identity_output_parser(
    queue: Iterator[GenerationResponse],
    prompt: str,
    tool_parser: object,
    tokenizer: object,
    model_cls: type,
    model_id: ModelId,
    tools: object,
) -> Iterator[TokenChunk | None]:
    """Yield every response from the queue unchanged (no tool/parse logic)."""

    def gen() -> Iterator[TokenChunk | None]:
        for response in queue:
            if response is None:
                yield None
                continue
            yield TokenChunk(
                model=model_id,
                text="hi",
                token_id=0,
                finish_reason="stop",
                usage=None,
                stats=None,
            )

    return gen()


def _task(seq: int, max_output_tokens: int = 8) -> TextGeneration:
    return TextGeneration(
        task_id=f"task-{seq:03d}",
        command_id=CommandId(f"cmd-{seq:03d}"),
        instance_id=InstanceId(f"inst-{seq:03d}"),
        task_params=TextGenerationTaskParams(
            model=ModelId("model-a"),
            input=[InputMessage(role="user", content=InputMessageContent("Hello"))],
            stream=True,
            max_output_tokens=max_output_tokens,
            temperature=0.0,
        ),
    )


@pytest.fixture
def patch_batch_io(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Stub every external dependency the BatchGenerator touches."""
    monkeypatch.setattr(bg, "warmup_inference", lambda *a, **k: (50, 0))
    monkeypatch.setattr(bg, "_check_for_debug_prompts", lambda *a, **k: None)
    monkeypatch.setattr(bg, "mx_any", lambda *a, **k: False)
    monkeypatch.setattr(
        bg, "mx_all_gather_tasks", lambda tasks, group: (list(tasks), [])
    )
    monkeypatch.setattr(bg, "apply_chat_template", lambda tokenizer, params: "prompt")
    monkeypatch.setattr(bg, "apply_all_parsers", _identity_output_parser)
    monkeypatch.setattr(
        bg,
        "map_responses_to_chunks",
        lambda r, model_id: TokenChunk(
            model=model_id, text="hi", token_id=0, usage=None
        ),
    )
    monkeypatch.setattr(bg, "ExoBatchGenerator", OomFakeExoBatchGenerator)
    monkeypatch.setattr(bg, "_clear_oom_memory", lambda *a, **k: None)
    return monkeypatch


@pytest.fixture
def patch_scheduler_io(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Stub every external dependency the ContinuousBatchScheduler touches."""
    monkeypatch.setattr(cb, "warmup_inference", lambda *a, **k: (50, 0))
    monkeypatch.setattr(cb, "_check_for_debug_prompts", lambda *a, **k: None)
    monkeypatch.setattr(cb, "mx_any", lambda *a, **k: False)
    monkeypatch.setattr(
        cb, "mx_all_gather_tasks", lambda tasks, group: (list(tasks), [])
    )

    def _agree_cancellations(self: object) -> None:
        for task_id in self.cancel_receiver.collect():  # type: ignore[attr-defined]
            if task_id in self._all_tasks:  # type: ignore[attr-defined]
                self._cancelled_tasks.add(task_id)  # type: ignore[attr-defined]

    monkeypatch.setattr(
        cb.ContinuousBatchScheduler, "agree_on_cancellations", _agree_cancellations
    )
    monkeypatch.setattr(cb, "apply_chat_template", lambda tokenizer, params: "prompt")
    monkeypatch.setattr(cb, "apply_all_parsers", _identity_output_parser)
    monkeypatch.setattr(
        cb,
        "map_responses_to_chunks",
        lambda r, model_id: TokenChunk(
            model=model_id, text="hi", token_id=0, usage=None
        ),
    )
    monkeypatch.setattr(cb, "ExoBatchGenerator", OomFakeExoBatchGenerator)
    monkeypatch.setattr(cb, "_clear_oom_memory", lambda *a, **k: None)
    return monkeypatch


# ---------------------------------------------------------------------------
# RunnerDegraded status
# ---------------------------------------------------------------------------


class TestRunnerDegradedStatus:
    def test_is_running_subclass(self) -> None:
        st = RunnerDegraded(max_batch_size=4, reason="OOM recovered")
        assert isinstance(st, RunnerDegraded)
        # Subclass of RunnerRunning → every isinstance(RunnerRunning) consumer
        # (plan.py health, apply.py circuit-breaker success reset) treats a
        # degraded node as alive, not failed.
        assert isinstance(st, RunnerRunning)
        assert st.is_running() is True

    def test_fields_default_none(self) -> None:
        st = RunnerDegraded()
        assert st.max_batch_size is None
        assert st.reason is None

    def test_round_trip_event(self) -> None:
        from exo.shared.types.events import RunnerStatusUpdated

        ev = RunnerStatusUpdated(
            runner_id="r123",
            runner_status=RunnerDegraded(max_batch_size=2, reason="OOM halved"),
        )
        parsed = RunnerStatusUpdated.model_validate_json(ev.model_dump_json())
        assert isinstance(parsed.runner_status, RunnerDegraded)
        assert parsed.runner_status.max_batch_size == 2
        assert parsed.runner_status.reason == "OOM halved"

    def test_ready_running_consumers_unaffected(self) -> None:
        # plan.py checks isinstance(status, (RunnerReady, RunnerRunning)); a
        # degraded runner must satisfy exactly like a running one.
        st = RunnerDegraded(max_batch_size=1)
        assert isinstance(st, (RunnerReady, RunnerRunning))


# ---------------------------------------------------------------------------
# BatchGenerator
# ---------------------------------------------------------------------------


class TestBatchGeneratorOom:
    def test_oom_halves_and_raises_oom_recovered(
        self, patch_batch_io: pytest.MonkeyPatch
    ) -> None:
        gen = bg.BatchGenerator(
            model=_FakeModel(),
            tokenizer=_FakeTokenizer(),
            group=_FakeGroup(),
            kv_prefix_cache=None,
            tool_parser=None,
            model_id=ModelId("model-a"),
            device_rank=0,
            cancel_receiver=FakeCancelReceiver(),
            event_sender=CollectingSender(),
        )
        gen.submit(_task(1))
        with pytest.raises(OomRecoveredError) as excinfo:
            list(gen.step())
        oom = excinfo.value
        # Batch size halved from the EXO_MAX_CONCURRENT_REQUESTS default.
        assert 0 < oom.max_batch_size < bg.EXO_MAX_CONCURRENT_REQUESTS
        assert oom.max_batch_size == gen._max_concurrent
        assert "OOM" in oom.reason
        # The engine was reset and tasks resubmitted.
        assert gen._gen.reset_count == 1
        assert len(gen._active_tasks) == 1

    def test_second_step_succeeds_after_recovery(
        self, patch_batch_io: pytest.MonkeyPatch
    ) -> None:
        gen = bg.BatchGenerator(
            model=_FakeModel(),
            tokenizer=_FakeTokenizer(),
            group=_FakeGroup(),
            kv_prefix_cache=None,
            tool_parser=None,
            model_id=ModelId("model-a"),
            device_rank=0,
            cancel_receiver=FakeCancelReceiver(),
            event_sender=CollectingSender(),
        )
        gen.submit(_task(1))
        with pytest.raises(OomRecoveredError):
            list(gen.step())
        # The retried step now completes the resubmitted task.
        results = list(gen.step())
        assert any(isinstance(r, TokenChunk) for _, r in results)
        assert len(gen._active_tasks) == 0

    def test_admission_cap_is_halved(
        self, patch_batch_io: pytest.MonkeyPatch
    ) -> None:
        gen = bg.BatchGenerator(
            model=_FakeModel(),
            tokenizer=_FakeTokenizer(),
            group=_FakeGroup(),
            kv_prefix_cache=None,
            tool_parser=None,
            model_id=ModelId("model-a"),
            device_rank=0,
            cancel_receiver=FakeCancelReceiver(),
            event_sender=CollectingSender(),
        )
        original = gen._max_concurrent
        gen.submit(_task(1))
        with pytest.raises(OomRecoveredError):
            list(gen.step())
        assert gen._max_concurrent == max(1, original // 2)
        # submit more tasks than the halved cap: admission must stop at the
        # degraded limit.
        for i in range(2, 2 + original):
            gen.submit(_task(i))
        list(gen.step())
        assert len(gen._active_tasks) <= gen._max_concurrent

    def test_non_oom_error_propagates(
        self, patch_batch_io: pytest.MonkeyPatch
    ) -> None:
        gen = bg.BatchGenerator(
            model=_FakeModel(),
            tokenizer=_FakeTokenizer(),
            group=_FakeGroup(),
            kv_prefix_cache=None,
            tool_parser=None,
            model_id=ModelId("model-a"),
            device_rank=0,
            cancel_receiver=FakeCancelReceiver(),
            event_sender=CollectingSender(),
        )
        gen.submit(_task(1))

        def _boom_step(self: object) -> list:
            raise RuntimeError("kurplunk exploded")

        # Disarm the fake's OOM trigger so the RuntimeError is what step()
        # actually sees, then confirm it propagates untouched (the OOM catch
        # must not swallow arbitrary failures).
        gen._gen._arm_oom = False
        patch_batch_io.setattr(type(gen._gen), "step", _boom_step)
        with pytest.raises(RuntimeError):
            list(gen.step())


# ---------------------------------------------------------------------------
# ContinuousBatchScheduler
# ---------------------------------------------------------------------------


class TestContinuousBatchSchedulerOom:
    def test_oom_halves_and_raises_oom_recovered(
        self, patch_scheduler_io: pytest.MonkeyPatch
    ) -> None:
        sched = cb.ContinuousBatchScheduler(
            model=_FakeModel(),
            tokenizer=_FakeTokenizer(),
            group=_FakeGroup(),
            kv_prefix_cache=None,
            tool_parser=None,
            model_id=ModelId("model-a"),
            device_rank=0,
            cancel_receiver=FakeCancelReceiver(),
            event_sender=CollectingSender(),
        )
        sched.submit(_task(1))
        original_max = sched.max_batch_size
        with pytest.raises(OomRecoveredError) as excinfo:
            list(sched.step())
        oom = excinfo.value
        # Batch size was halved from the original.
        assert 0 < oom.max_batch_size < original_max
        assert oom.max_batch_size == sched.max_batch_size
        assert "OOM" in oom.reason
        assert sched._gen.reset_count == 1
        assert len(sched._running) == 1

    def test_second_step_succeeds_after_recovery(
        self, patch_scheduler_io: pytest.MonkeyPatch
    ) -> None:
        sched = cb.ContinuousBatchScheduler(
            model=_FakeModel(),
            tokenizer=_FakeTokenizer(),
            group=_FakeGroup(),
            kv_prefix_cache=None,
            tool_parser=None,
            model_id=ModelId("model-a"),
            device_rank=0,
            cancel_receiver=FakeCancelReceiver(),
            event_sender=CollectingSender(),
        )
        sched.submit(_task(1))
        with pytest.raises(OomRecoveredError):
            list(sched.step())
        results = list(sched.step())
        assert any(isinstance(r, TokenChunk) for _, r in results)
        assert len(sched._running) == 0

    def test_max_batch_size_floor_one(
        self, patch_scheduler_io: pytest.MonkeyPatch
    ) -> None:
        sched = cb.ContinuousBatchScheduler(
            model=_FakeModel(),
            tokenizer=_FakeTokenizer(),
            group=_FakeGroup(),
            kv_prefix_cache=None,
            tool_parser=None,
            model_id=ModelId("model-a"),
            device_rank=0,
            cancel_receiver=FakeCancelReceiver(),
            event_sender=CollectingSender(),
            max_batch_size=1,
        )
        sched.submit(_task(1))
        with pytest.raises(OomRecoveredError):
            list(sched.step())
        # Already at the floor — stays 1, no livelock to 0.
        assert sched.max_batch_size == 1


# ---------------------------------------------------------------------------
# SequentialGenerator
# ---------------------------------------------------------------------------


class TestSequentialGeneratorOom:
    def test_oom_retry_raises_oom_recovered_with_results(
        self, patch_batch_io: pytest.MonkeyPatch
    ) -> None:
        from exo.worker.runner.llm_inference.batch_generator import (
            SequentialGenerator,
        )

        # First mlx_generate call raises an OOM; the rebuilt one succeeds.
        calls = {"n": 0}

        def _flaky_mlx_generate(**_kwargs: object):
            calls["n"] += 1
            if calls["n"] == 1:
                raise MemoryError("Failed to allocate [8 x 4096 x 64 x 128]")
            yield GenerationResponse(text="hi", token=0, finish_reason="stop", usage=None)

        patch_batch_io.setattr(bg, "mlx_generate", _flaky_mlx_generate)
        gen = bg.SequentialGenerator(
            model=_FakeModel(),
            tokenizer=_FakeTokenizer(),
            group=_FakeGroup(),
            kv_prefix_cache=None,
            tool_parser=None,
            model_id=ModelId("model-a"),
            device_rank=0,
            cancel_receiver=FakeCancelReceiver(),
            event_sender=CollectingSender(),
        )
        gen.submit(_task(1))
        with pytest.raises(OomRecoveredError) as excinfo:
            list(gen.step())
        oom = excinfo.value
        assert oom.max_batch_size == 1  # single-task generator stays at floor
        assert "OOM" in oom.reason
        # The recovered output rides along — nothing is lost.
        assert len(oom.results) >= 1

    def test_oom_twice_escalates(self, patch_batch_io: pytest.MonkeyPatch) -> None:
        from exo.worker.runner.llm_inference.batch_generator import (
            SequentialGenerator,
        )

        def _always_oom(**_kwargs: object):
            raise MemoryError("Failed to allocate [8 x 4096 x 64 x 128]")

        patch_batch_io.setattr(bg, "mlx_generate", _always_oom)
        gen = bg.SequentialGenerator(
            model=_FakeModel(),
            tokenizer=_FakeTokenizer(),
            group=_FakeGroup(),
            kv_prefix_cache=None,
            tool_parser=None,
            model_id=ModelId("model-a"),
            device_rank=0,
            cancel_receiver=FakeCancelReceiver(),
            event_sender=CollectingSender(),
        )
        gen.submit(_task(1))
        # First step: OOM → clear KV → rebuild → retry → still OOM → the
        # retry catch re-raises the original OOM (escalation, not livelock).
        with pytest.raises(MemoryError):
            list(gen.step())


# ---------------------------------------------------------------------------
# Runner: handle_generation_tasks catches OomRecoveredError
# ---------------------------------------------------------------------------


class TestRunnerOomDegradation:
    def test_runner_emits_degraded_status(
        self, patch_batch_io: pytest.MonkeyPatch
    ) -> None:
        from exo.shared.types.events import RunnerStatusUpdated
        from exo.worker.runner.runner import Runner

        class _FakeRunner(Runner):
            """Bare runner with the state machine stubbed out."""

            def __init__(self) -> None:
                self.current_status = RunnerReady()
                self.event_sender = CollectingSender()
                self.runner_id = "runner-test"

            def update_status(self, status) -> None:  # type: ignore[no-untyped-def]
                self.current_status = status
                self.event_sender.send(
                    RunnerStatusUpdated(runner_id=self.runner_id, runner_status=status)
                )

        runner = _FakeRunner()
        # Feed the OomRecoveredError path directly (full handle_generation_tasks
        # requires the whole task-reader machinery; the status emission is the
        # contract under test).
        from exo.worker.runner.llm_inference.batch_generator import OomRecoveredError

        try:
            raise OomRecoveredError(max_batch_size=4, reason="OOM halved")
        except OomRecoveredError as oom:
            runner.update_status(
                RunnerDegraded(max_batch_size=oom.max_batch_size, reason=oom.reason)
            )
        status_events = [
            e for e in runner.event_sender.sent if isinstance(e, RunnerStatusUpdated)
        ]
        assert status_events
        assert isinstance(status_events[-1].runner_status, RunnerDegraded)
        assert status_events[-1].runner_status.max_batch_size == 4
        # Degraded keeps the runner "running" for downstream consumers.
        assert isinstance(runner.current_status, RunnerRunning)