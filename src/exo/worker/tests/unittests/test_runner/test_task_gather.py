# pyright: reportPrivateUsage=false
"""The task agreement split into a queued count gather and a later read, and the
batch engine's overlapped use of it during decode."""

from collections import deque
from dataclasses import dataclass, field

import mlx.core as mx
import pytest

import exo.worker.runner.llm_inference.batch_generator as mlx_batch_generator
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.worker.engines.mlx.utils_mlx import (
    TaskGather,
    finish_task_gather,
    mx_all_gather_tasks,
    start_task_gather,
)
from exo.worker.runner.llm_inference.batch_generator import BatchGenerator


@dataclass
class _Task:
    task_id: TaskId


def _task(n: int) -> TextGeneration:
    return _Task(TaskId(f"00000000-0000-0000-0000-00000000000{n}"))  # pyright: ignore[reportReturnType]


def test_single_rank_start_then_finish_matches_the_waited_gather() -> None:
    tasks = [_task(2), _task(1)]
    assert finish_task_gather(start_task_gather(tasks, None), None) == (
        mx_all_gather_tasks(tasks, None)
    )
    assert finish_task_gather(start_task_gather([], None), None) == ([], [])


@dataclass
class _Engine:
    """The fields BatchGenerator's agreement methods touch."""

    group: object = None
    _maybe_queue: list[TextGeneration] = field(default_factory=list)
    _queue: deque[TextGeneration] = field(default_factory=deque)
    _task_gather: TaskGather | None = None

    # The agreement methods under test, run against these fields only.
    agree_on_tasks = BatchGenerator.agree_on_tasks
    _overlap_agreement = BatchGenerator._overlap_agreement
    _finish_task_gather = BatchGenerator._finish_task_gather


@dataclass
class _Gen:
    """Stands in for ``ExoBatchGenerator``: decodes nothing, records no work."""

    _has_work: bool = False

    @property
    def has_work(self) -> bool:
        return self._has_work

    def step(self) -> list[tuple[int, object]]:
        return []

    def cancel(self, uids: list[int]) -> None:
        pass


@dataclass
class _StepEngine:
    """Enough of ``BatchGenerator`` to exercise the agreement dispatch at the top
    of ``step``: the two agreement methods record which path was taken, then raise
    :class:`_DispatchedError` to end the step. The decode path below the dispatch
    is not what is under test, so submitting a queued task also raises it.
    """

    group: object = None
    has_work: bool = False
    took: list[str] = field(default_factory=list)
    _maybe_queue: list[TextGeneration] = field(default_factory=list)
    _queue: deque[TextGeneration] = field(default_factory=deque)
    _task_gather: TaskGather | None = None
    # Read by ``step``'s submit loop before it calls the stubbed ``_start_task``.
    _active_tasks: dict[int, object] = field(default_factory=dict)

    @property
    def _gen(self) -> _Gen:
        return _Gen(_has_work=self.has_work)

    def agree_on_tasks(self) -> None:
        self.took.append("waited")
        raise _DispatchedError

    def _overlap_agreement(self) -> None:
        self.took.append("overlapped")
        raise _DispatchedError

    def _start_task(self, task: TextGeneration) -> int:
        # ``step`` catches this, calls ``_send_error``, then re-raises it.
        raise _DispatchedError

    def _send_error(self, task: TextGeneration, e: Exception) -> None:
        pass

    step = BatchGenerator.step


class _DispatchedError(Exception):
    """Raised by a stub to end the step once the path under test is reached."""


def _run_step(engine: _StepEngine) -> str:
    """Run one step, returning the agreement path it dispatched to."""
    engine.took.clear()
    with pytest.raises(_DispatchedError):
        list(engine.step())
    assert len(engine.took) == 1, engine.took
    return engine.took[0]


@pytest.fixture
def one_rank_gathers(monkeypatch: pytest.MonkeyPatch) -> None:
    def start(tasks: list[TextGeneration], group: object) -> TaskGather:
        return TaskGather(tasks=list(tasks), counts=mx.array([len(tasks)]))

    def finish(
        gather: TaskGather, group: object
    ) -> tuple[list[TextGeneration], list[TextGeneration]]:
        return sorted(gather.tasks, key=lambda task: task.task_id), []

    monkeypatch.setattr(mlx_batch_generator, "start_task_gather", start)
    monkeypatch.setattr(mlx_batch_generator, "finish_task_gather", finish)

    def gather(
        tasks: list[TextGeneration], group: object
    ) -> tuple[list[TextGeneration], list[TextGeneration]]:
        return finish(start(tasks, group), group)

    monkeypatch.setattr(mlx_batch_generator, "mx_all_gather_tasks", gather)


@pytest.mark.usefixtures("one_rank_gathers")
def test_overlap_agrees_one_step_later_and_keeps_late_tasks_pending() -> None:
    engine = _Engine()
    first, late = _task(1), _task(2)
    engine._maybe_queue.append(first)
    engine._overlap_agreement()
    assert list(engine._queue) == []  # started, not applied yet
    engine._maybe_queue.append(late)  # arrives while the gather is in flight
    engine._overlap_agreement()
    assert list(engine._queue) == [first]
    assert engine._maybe_queue == [late]  # not in the finished gather: stays pending
    engine._overlap_agreement()
    assert list(engine._queue) == [first, late]


@pytest.mark.usefixtures("one_rank_gathers")
def test_waited_agreement_finishes_the_overlapped_one_first() -> None:
    engine = _Engine()
    first, second = _task(1), _task(2)
    engine._maybe_queue.append(first)
    engine._overlap_agreement()
    engine._maybe_queue.append(second)
    engine.agree_on_tasks()
    assert list(engine._queue) == [first, second]
    assert engine._task_gather is None
    assert engine._maybe_queue == []


@pytest.mark.usefixtures("one_rank_gathers")
def test_step_overlaps_the_agreement_only_while_the_engine_has_work() -> None:
    """``step`` must take the overlapped path while decoding and the waited one
    when idle: the point of the split is that a decode step never blocks on the
    count gather."""
    # While decoding: a step overlaps, leaving the gather in flight.
    assert _run_step(_StepEngine(has_work=True)) == "overlapped"
    # Idle: a step waits, so nothing is left in flight.
    assert _run_step(_StepEngine(has_work=False)) == "waited"
    # Idle but with tasks already queued: the queue is non-empty, so the step
    # skips agreement entirely and goes straight to submitting them.
    queued = _StepEngine(has_work=False)
    queued._queue.append(_task(1))
    queued.took.append("noop")
    with pytest.raises(_DispatchedError):  # reached by submitting the queued task
        list(queued.step())
    assert queued.took == ["noop"], "agreement ran despite a non-empty queue"
