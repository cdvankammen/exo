import time
from collections.abc import Callable
from typing import cast

import anyio
import pytest

from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
)
from exo.shared.types.tasks import Task, TaskId, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerIdle,
    RunnerRunning,
    RunnerWarmingUp,
)
from exo.utils.async_process import AsyncProcess
from exo.utils.channels import Sender, channel, mp_channel
from exo.worker.runner.bootstrap import RunnerTerminationError
from exo.worker.runner.supervisor import (
    TASK_ACK_TIMEOUT_SECONDS,
    RunnerStdioHandler,
    RunnerSupervisor,
)
from exo.worker.tests.unittests.conftest import get_bound_mlx_ring_instance


def _sleep_forever(*_args: object) -> None:
    time.sleep(1000)


class _DeadProcess:
    def __init__(self):
        rx1, _ = channel[bytes]()
        rx2, _ = channel[bytes]()
        self.stdout = rx1
        self.stderr = rx2

    exitcode = -6

    def is_alive(self) -> bool:
        return False


@pytest.mark.anyio
async def test_check_runner_emits_error_chunk_for_inflight_text_generation() -> None:
    event_sender, event_receiver = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event | RunnerTerminationError]()

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )

    proc = cast(AsyncProcess, cast(object, _DeadProcess()))
    handler = await RunnerStdioHandler.create(
        stdout_rx=proc.stdout, stderr_rx=proc.stderr
    )
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=proc,
        _runner_stdio_handler=handler,
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )

    command_id = CommandId("cmd-a")
    task = TextGeneration(
        task_id=TaskId("task-a"),
        instance_id=bound_instance.instance.instance_id,
        command_id=command_id,
        task_params=TextGenerationTaskParams(
            model=bound_instance.bound_shard.model_card.model_id,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
            stream=True,
        ),
    )
    supervisor.in_progress[task.task_id] = task
    supervisor.shutdown = lambda: None

    await supervisor._check_runner(RuntimeError("boom"))  # pyright: ignore[reportPrivateUsage]

    got_chunk = await event_receiver.receive()
    got_status = await event_receiver.receive()

    assert isinstance(got_chunk, ChunkGenerated)
    assert got_chunk.command_id == command_id
    assert isinstance(got_chunk.chunk, ErrorChunk)
    assert "Runner shutdown before completing command" in got_chunk.chunk.error_message

    assert isinstance(got_status, RunnerStatusUpdated)
    assert isinstance(got_status.runner_status, RunnerFailed)

    event_sender.close()
    with anyio.move_on_after(0.1):
        await event_receiver.aclose()


@pytest.mark.anyio
async def test_wait_stopped_resolves_only_after_process_actually_exits() -> None:
    """Regression test: main.py's Shutdown handling awaits wait_stopped()
    before the next plan() tick is allowed to create a replacement runner for
    the same instance. If wait_stopped() resolved before the OS process (and
    whatever resources it held, e.g. an RDMA queue pair) actually went away,
    a fast Shutdown->CreateRunner cycle could race the old process's
    teardown."""
    event_sender, event_receiver = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event | RunnerTerminationError]()

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )

    runner_process = AsyncProcess(target=_sleep_forever, args=(), daemon=True)
    handler = await RunnerStdioHandler.create(
        stdout_rx=runner_process.stdout, stderr_rx=runner_process.stderr
    )
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=runner_process,
        _runner_stdio_handler=handler,
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )

    async with anyio.create_task_group() as tg:
        tg.start_soon(supervisor.run)

        with anyio.fail_after(5):
            while not runner_process.is_alive():
                await anyio.sleep(0.01)

        assert not supervisor._stopped.is_set()  # pyright: ignore[reportPrivateUsage]

        supervisor.shutdown()

        with anyio.fail_after(10):
            await supervisor.wait_stopped()

        assert not runner_process.is_alive()

        # Safe to await again once already stopped (level-triggered event).
        with anyio.fail_after(1):
            await supervisor.wait_stopped()

    event_sender.close()
    with anyio.move_on_after(0.1):
        await event_receiver.aclose()


@pytest.mark.anyio
async def test_duplicate_task_acknowledged_does_not_crash_forward_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RC-05: a duplicate/stale TaskAcknowledged must not KeyError and kill
    _forward_events — the supervisor must keep running and forwarding events."""
    monkeypatch.setattr(
        "exo.worker.runner.supervisor.TASK_ACK_TIMEOUT_SECONDS", 0.5
    )
    event_sender, event_receiver = channel[Event]()
    ev_send, ev_recv = mp_channel[Event | RunnerTerminationError]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()

    bound_instance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-dup"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-dup"),
        node_id=NodeId("node-dup"),
    )

    runner_process = AsyncProcess(target=_sleep_forever, args=(), daemon=True)
    handler = await RunnerStdioHandler.create(
        stdout_rx=runner_process.stdout, stderr_rx=runner_process.stderr
    )
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=runner_process,
        _runner_stdio_handler=handler,
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )

    async with anyio.create_task_group() as tg:
        tg.start_soon(supervisor.run)

        # Wait for the supervisor's runner process to come alive.
        with anyio.fail_after(5):
            while not runner_process.is_alive():
                await anyio.sleep(0.01)

        task_id = TaskId("task-dup-1")
        event = anyio.Event()
        supervisor.pending[task_id] = event
        supervisor.in_progress[task_id] = TextGeneration(
            task_id=task_id,
            instance_id=bound_instance.instance.instance_id,
            command_id=CommandId("cmd-dup"),
            task_params=TextGenerationTaskParams(
                model=bound_instance.bound_shard.model_card.model_id,
                input=[InputMessage(role="user", content=InputMessageContent("hi"))],
                stream=True,
            ),
        )

        # Inject two TaskAcknowledged for the same id; the second is the
        # race condition that previously raised KeyError.  Use synchronous
        # send() to avoid deadlocking on the module-level async limiter
        # shared with _forward_events's blocking receive_async.
        ev_send.send(TaskAcknowledged(task_id=task_id))
        ev_send.send(TaskAcknowledged(task_id=task_id))
        with anyio.fail_after(3):
            await event.wait()

        # _forward_events must still be alive and forwarding: a follow-up
        # event injected after the duplicate must arrive at event_receiver.
        followup = RunnerStatusUpdated(
            runner_id=bound_instance.bound_runner_id,
            runner_status=RunnerIdle(),
        )
        ev_send.send(followup)
        with anyio.fail_after(3):
            forwarded = await event_receiver.receive()
        assert isinstance(forwarded, RunnerStatusUpdated)

        supervisor.shutdown()

    event_sender.close()
    ev_send.close()
    with anyio.move_on_after(0.1):
        await event_receiver.aclose()


@pytest.mark.anyio
async def test_start_task_cleans_pending_on_closed_resource() -> None:
    """RC-05: when send_async raises ClosedResourceError, pending and
    in_progress must both be cleared and start_task must return promptly."""
    event_sender, event_receiver = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event | RunnerTerminationError]()

    bound_instance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-closed"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-closed"),
        node_id=NodeId("node-closed"),
    )

    runner_process = AsyncProcess(target=_sleep_forever, args=(), daemon=True)
    handler = await RunnerStdioHandler.create(
        stdout_rx=runner_process.stdout, stderr_rx=runner_process.stderr
    )
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=runner_process,
        _runner_stdio_handler=handler,
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )

    task = TextGeneration(
        task_id=TaskId("task-closed"),
        instance_id=bound_instance.instance.instance_id,
        command_id=CommandId("cmd-closed"),
        task_params=TextGenerationTaskParams(
            model=bound_instance.bound_shard.model_card.model_id,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
            stream=True,
        ),
    )

    # Close the sender so send_async raises ClosedResourceError.
    task_sender.close()
    with anyio.fail_after(3):
        await supervisor.start_task(task)

    assert task.task_id not in supervisor.pending
    assert task.task_id not in supervisor.in_progress

    event_sender.close()
    ev_recv.close()
    with anyio.move_on_after(0.1):
        await event_receiver.aclose()


@pytest.mark.anyio
async def test_start_task_returns_on_ack_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RC-05: if the runner never acknowledges, start_task must return after
    TASK_ACK_TIMEOUT_SECONDS instead of hanging forever."""
    monkeypatch.setattr(
        "exo.worker.runner.supervisor.TASK_ACK_TIMEOUT_SECONDS", 0.1
    )
    event_sender, event_receiver = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event | RunnerTerminationError]()

    bound_instance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-timeout"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-timeout"),
        node_id=NodeId("node-timeout"),
    )

    runner_process = AsyncProcess(target=_sleep_forever, args=(), daemon=True)
    handler = await RunnerStdioHandler.create(
        stdout_rx=runner_process.stdout, stderr_rx=runner_process.stderr
    )
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=runner_process,
        _runner_stdio_handler=handler,
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )

    task = TextGeneration(
        task_id=TaskId("task-timeout"),
        instance_id=bound_instance.instance.instance_id,
        command_id=CommandId("cmd-timeout"),
        task_params=TextGenerationTaskParams(
            model=bound_instance.bound_shard.model_card.model_id,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
            stream=True,
        ),
    )

    with anyio.fail_after(3):
        await supervisor.start_task(task)

    # The event was never set by _forward_events, so the timeout path must
    # have cleaned up both dicts.
    assert task.task_id not in supervisor.pending
    assert task.task_id not in supervisor.in_progress

    event_sender.close()
    ev_recv.close()
    with anyio.move_on_after(0.1):
        await event_receiver.aclose()


class _HookedEventSender:
    """Runs a hook before forwarding each event, standing in for
    _forward_events mutating in_progress while a send is in flight."""

    def __init__(self, inner: Sender[Event]) -> None:
        self.inner = inner
        self.on_send: Callable[[], object] = lambda: None

    async def send(self, event: Event) -> None:
        self.on_send()
        await self.inner.send(event)


def _text_generation(bound_instance: BoundInstance, name: str) -> TextGeneration:
    return TextGeneration(
        task_id=TaskId(f"task-{name}"),
        instance_id=bound_instance.instance.instance_id,
        command_id=CommandId(f"cmd-{name}"),
        task_params=TextGenerationTaskParams(
            model=bound_instance.bound_shard.model_card.model_id,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
            stream=True,
        ),
    )


@pytest.mark.anyio
async def test_check_runner_survives_task_completing_during_error_fanout() -> None:
    event_sender, event_receiver = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event | RunnerTerminationError]()

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )

    proc = cast(AsyncProcess, cast(object, _DeadProcess()))
    handler = await RunnerStdioHandler.create(
        stdout_rx=proc.stdout, stderr_rx=proc.stderr
    )
    hooked_sender = _HookedEventSender(event_sender)
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=proc,
        _runner_stdio_handler=handler,
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _task_sender=task_sender,
        _event_sender=cast(Sender[Event], cast(object, hooked_sender)),
        _cancel_sender=cancel_sender,
    )

    first = _text_generation(bound_instance, "a")
    second = _text_generation(bound_instance, "b")
    supervisor.in_progress[first.task_id] = first
    supervisor.in_progress[second.task_id] = second
    supervisor.shutdown = lambda: None
    # the second task completes while the first task's ErrorChunk is being sent
    hooked_sender.on_send = lambda: supervisor.in_progress.pop(second.task_id, None)

    await supervisor._check_runner(RuntimeError("boom"))  # pyright: ignore[reportPrivateUsage]

    got_chunk = await event_receiver.receive()
    got_status = await event_receiver.receive()

    assert isinstance(got_chunk, ChunkGenerated)
    assert got_chunk.command_id == first.command_id
    assert isinstance(got_status, RunnerStatusUpdated)
    assert isinstance(got_status.runner_status, RunnerFailed)

    event_sender.close()
    with anyio.move_on_after(0.1):
        await event_receiver.aclose()
