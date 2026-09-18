"""Unit tests for Worker node-lifecycle internals in ``exo.worker.main``.

Covers the two biggest untested surfaces identified by the coverage-gap
analysis (test-coverage-gap-analysis-2026-09-17.md §3/§4/§7.4):

- :meth:`Worker.plan_step` (worker/main.py:215-398) — the core planning loop:
  CreateRunner happy path, instance-retry exhaustion (DeleteInstance via
  command sender), Shutdown/CancelTask dispatch, DownloadModel found/missing
  branches, and plain task forwarding to the runner.
- :meth:`Worker._poll_connection_updates` (worker/main.py:420-490) — the
  topology edge driver that emitted the TopologyEdgeCreated events at the
  root of the election-flapping research: new-edge creation, latency-change
  replace (delete+create), and stale-edge deletion.

These tests run one poll/plan iteration at a time instead of spinning the
infinite ``while True`` loops: the task under test is spawned with anyio
memory channels, drives one iteration, and is cancelled once it blocks on the
next receive. No real sockets, subprocesses, or multiprocessing is used.

# pyright: reportPrivateUsage=false
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Callable

import anyio
import pytest

from exo.shared.constants import EXO_MAX_INSTANCE_RETRIES
from exo.shared.types.commands import (
    DeleteInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
    StartDownload,
)
from exo.shared.types.common import NodeId, SystemId
from exo.shared.types.events import (
    Event,
    NodeDownloadProgress,
    TaskCreated,
    TaskStatusUpdated,
    TopologyEdgeCreated,
    TopologyEdgeDeleted,
)
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import (
    NetworkInterfaceInfo,
    NodeIdentity,
    NodeNetworkInfo,
)
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    CancelTask,
    CreateRunner,
    DownloadModel,
    Shutdown,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.topology import Connection, SocketConnection
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerIdle, RunnerReady, RunnerStatus
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.keyed_backoff import KeyedBackoff
from exo.utils.task_group import TaskGroup
from exo.worker.main import Worker
from exo.worker.tests.constants import (
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
    NODE_B,
    RUNNER_1_ID,
    TASK_1_ID,
)
from exo.worker.tests.unittests.conftest import (
    FakeRunnerSupervisor,
    get_bound_mlx_ring_instance,
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

# Type alias for the installed check_reachable signature.
Reachable = tuple[str, NodeId, float]


class _RecordingRunner(FakeRunnerSupervisor):
    """Fake runner supervisor with in-memory run/shutdown/start_task.

    Mirrors the parts of RunnerSupervisor the Worker's plan_step touches:
    ``run`` (spawned by _create_supervisor), ``start_task`` / ``cancel_task``
    (task dispatch), ``shutdown`` + ``wait_stopped`` (Shutdown fast path).
    """

    def __init__(
        self,
        *args: Any,
        status: RunnerStatus | None = None,
        **kwargs: Any,
    ) -> None:
        if status is None:
            status = RunnerIdle()
        super().__init__(*args, status=status, **kwargs)
        self.started: list[Any] = []
        self.cancelled_tasks: list[Any] = []
        self.cancelled: set[Any] = set()
        self.shutdown_called = False
        self._stopped = anyio.Event()

    async def run(self) -> None:
        await anyio.sleep(3600)

    async def start_task(self, task: Any) -> Any:
        self.started.append(task)

    async def cancel_task(self, task_id: Any) -> Any:
        self.cancelled_tasks.append(task_id)

    def shutdown(self) -> None:
        self.shutdown_called = True
        self._stopped.set()

    async def wait_stopped(self) -> None:
        await self._stopped.wait()


def _make_text_task() -> TextGeneration:
    return TextGeneration(
        task_id=TASK_1_ID,
        instance_id=INSTANCE_1_ID,
        command_id=TASK_1_ID,
        task_params=TextGenerationTaskParams(
            model=MODEL_A_ID,
            input=[],
        ),
    )


def _make_single_node_bound_runner(
    instance_id: InstanceId = INSTANCE_1_ID,
    runner_id: RunnerId = RUNNER_1_ID,
) -> BoundInstance:
    """A bound instance with one node/one runner (no sibling runners)."""
    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=instance_id,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: runner_id},
        runner_to_shard={runner_id: shard},
    )
    return BoundInstance(
        instance=instance, bound_runner_id=runner_id, bound_node_id=NODE_A
    )


def _make_bound_runner(
    instance_id: InstanceId = INSTANCE_1_ID,
    runner_id: RunnerId = RUNNER_1_ID,
) -> BoundInstance:
    return get_bound_mlx_ring_instance(
        instance_id=instance_id,
        model_id=MODEL_A_ID,
        runner_id=runner_id,
        node_id=NODE_A,
    )


class _FakeSupervisor(_RecordingRunner):
    """Stand-in for RunnerSupervisor.create: never spawns subprocesses."""

    @classmethod
    async def create(cls, *args: Any, **kwargs: Any) -> "_FakeSupervisor":
        bound = kwargs.get("bound_instance")
        assert bound is not None
        return cls(bound_instance=bound, status=RunnerIdle())


class _Harness:
    """A Worker wired to real in-memory channels, bypassing __init__."""

    def __init__(
        self,
        runner: _RecordingRunner | None = None,
        state: State | None = None,
    ) -> None:
        event_sender, event_receiver = channel[Event]()
        command_sender, command_receiver = channel[ForwarderCommand]()
        download_sender, download_receiver = channel[ForwarderDownloadCommand]()

        worker = object.__new__(Worker)
        worker.node_id = NODE_A
        worker.event_receiver = event_receiver  # type: ignore[assignment]
        worker.event_sender = event_sender
        worker.command_sender = command_sender
        worker.download_command_sender = download_sender
        worker.api_port = 4001
        worker.no_downloads = False
        worker.state = state if state is not None else State()
        worker.runners = {}
        worker._system_id = SystemId()
        worker.input_chunk_buffer = {}
        worker.input_chunk_counts = {}
        worker.image_cache = {}
        worker._instance_backoff = KeyedBackoff[InstanceId]()
        worker._download_backoff = KeyedBackoff[str]()
        worker._edge_probe_failures = {}
        worker._stopped = anyio.Event()
        worker._tg = TaskGroup()
        # Install the fake supervisor factory so _create_supervisor never
        # spawns a real subprocess in tests.
        worker.supervisor_factory = _FakeSupervisor.create  # type: ignore[attr-defined]

        self.worker = worker
        self.event_sender = event_sender
        self.event_receiver = event_receiver
        self.events: list[Event] = []
        self.commands: list[ForwarderCommand] = []
        self.download_commands: list[ForwarderDownloadCommand] = []

        async def _drain_commands() -> None:
            with command_receiver:
                async for cmd in command_receiver:
                    self.commands.append(cmd)

        async def _drain_downloads() -> None:
            with download_receiver:
                async for cmd in download_receiver:
                    self.download_commands.append(cmd)

        self._drain_commands = _drain_commands
        self._drain_downloads = _drain_downloads

        if runner is not None:
            worker.runners[runner.bound_instance.bound_runner_id] = runner  # type: ignore[index]

    async def run_plan_once(self) -> list[Event]:
        """Run exactly one plan_step iteration; return all emitted events."""

        async def _collect() -> None:
            with self.event_receiver:
                async for ev in self.event_receiver:
                    self.events.append(ev)

        async with anyio.create_task_group() as tg:
            tg.start_soon(_collect)
            tg.start_soon(self._drain_commands)
            tg.start_soon(self._drain_downloads)
            await anyio.sleep(0)
            async with self.worker._tg:
                tg.start_soon(self.worker.plan_step)
                await anyio.sleep(0.2)
                tg.cancel_scope.cancel()
        await self.event_sender.aclose()
        return list(self.events)


# ---------------------------------------------------------------------------
# plan_step: CreateRunner paths
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_plan_step_create_runner_happy_path() -> None:
    """A pending instance without a local runner produces a CreateRunner task,
    creates a new supervisor via the fake factory, and emits TaskCreated +
    TaskStatusUpdated(Complete)."""
    import exo.worker.main as worker_main
    from exo.worker.runner.supervisor import RunnerSupervisor

    harness = _Harness()
    bound = _make_bound_runner()
    harness.worker.state = State(
        instances={INSTANCE_1_ID: bound.instance},
        runners={RUNNER_1_ID: RunnerIdle()},
    )
    original_create = RunnerSupervisor.create
    RunnerSupervisor.create = _FakeSupervisor.create  # type: ignore[method-assign]
    try:
        events = await harness.run_plan_once()
    finally:
        RunnerSupervisor.create = original_create  # type: ignore[method-assign]

    assert RUNNER_1_ID in harness.worker.runners
    assert isinstance(harness.worker.runners[RUNNER_1_ID], _FakeSupervisor)
    created = [ev for ev in events if isinstance(ev, TaskCreated)]
    assert len(created) == 1
    assert isinstance(created[0].task, CreateRunner)
    assert created[0].task.instance_id == INSTANCE_1_ID
    completed = [ev for ev in events if isinstance(ev, TaskStatusUpdated)]
    assert any(ev.task_status == TaskStatus.Complete for ev in completed)


@pytest.mark.anyio
async def test_plan_step_create_runner_retry_exhaustion_requests_deletion() -> None:
    """Once instance retries hit EXO_MAX_INSTANCE_RETRIES, plan_step stops
    creating the runner and requests DeleteInstance via the command channel."""
    bound = _make_bound_runner()
    harness = _Harness(
        state=State(
            instances={INSTANCE_1_ID: bound.instance},
            runners={RUNNER_1_ID: RunnerIdle()},
        )
    )
    backoff = harness.worker._instance_backoff
    import time as _time

    # Record MAX retries, then age the last-attempt timestamp so
    # should_proceed() passes (backoff expired) while attempts remain >= MAX.
    for _ in range(EXO_MAX_INSTANCE_RETRIES):
        backoff.record_attempt(INSTANCE_1_ID)
    backoff._last_time[INSTANCE_1_ID] = _time.monotonic() - 60.0

    events = await harness.run_plan_once()

    assert RUNNER_1_ID not in harness.worker.runners
    delete_cmds = [
        cmd
        for cmd in harness.commands
        if isinstance(cmd.command, DeleteInstance)
    ]
    assert len(delete_cmds) == 1
    assert delete_cmds[0].command.instance_id == INSTANCE_1_ID  # type: ignore[union-attr]


@pytest.mark.anyio
async def test_plan_step_returns_no_task_when_nothing_to_do() -> None:
    """With no pending work, plan_step never emits anything."""
    harness = _Harness()
    harness.worker.state = State()

    events = await harness.run_plan_once()

    assert events == []


# ---------------------------------------------------------------------------
# plan_step: Shutdown / CancelTask dispatch
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_plan_step_shutdown_fast_path() -> None:
    """A Shutdown task pops the runner and forwards it to start_task without
    creating a new supervisor or waiting on a real runner process."""
    bound = _make_bound_runner()
    runner = _RecordingRunner(bound_instance=bound)
    harness = _Harness(
        runner=runner,
        state=State(
            instances={},  # instance gone → _kill_runner emits Shutdown
            runners={RUNNER_1_ID: RunnerIdle()},
        ),
    )

    events = await harness.run_plan_once()

    assert RUNNER_1_ID not in harness.worker.runners
    assert any(isinstance(t, Shutdown) for t in runner.started)


@pytest.mark.anyio
async def test_plan_step_cancel_task_dispatch() -> None:
    """A Cancelled task produces a CancelTask routed to the right runner."""
    bound = _make_bound_runner()
    runner = _RecordingRunner(bound_instance=bound)
    text_task = _make_text_task().model_copy(update={"task_status": TaskStatus.Cancelled})
    harness = _Harness(
        runner=runner,
        state=State(
            instances={INSTANCE_1_ID: bound.instance},
            runners={RUNNER_1_ID: RunnerIdle()},
            tasks={TASK_1_ID: text_task},
        ),
    )

    events = await harness.run_plan_once()

    assert runner.cancelled_tasks == [TASK_1_ID]
    completed = [ev for ev in events if isinstance(ev, TaskStatusUpdated)]
    assert any(ev.task_status == TaskStatus.Complete for ev in completed)


# ---------------------------------------------------------------------------
# plan_step: DownloadModel branches
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_plan_step_download_model_missing_forwards_start_download() -> None:
    """When the model shard is absent, plan_step emits StartDownload via the
    download-command channel and marks the task Running."""
    import exo.worker.main as worker_main

    def _no_resolve(*_args: Any, **_kwargs: Any) -> Any:
        return None

    worker_main.resolve_existing_model = _no_resolve  # type: ignore[assignment]
    worker_main.is_read_only_model_dir = lambda _p: False  # type: ignore[assignment]

    bound = _make_bound_runner()
    harness = _Harness(
        runner=_RecordingRunner(bound_instance=bound),
        state=State(
            instances={INSTANCE_1_ID: bound.instance},
            runners={RUNNER_1_ID: RunnerIdle()},
            downloads={NODE_A: []},
        ),
    )

    events = await harness.run_plan_once()

    assert len(harness.download_commands) == 1
    assert isinstance(harness.download_commands[0].command, StartDownload)
    running = [ev for ev in events if isinstance(ev, TaskStatusUpdated)]
    assert any(ev.task_status == TaskStatus.Running for ev in running)


@pytest.mark.anyio
async def test_plan_step_download_model_found_emits_complete() -> None:
    """When a model already exists on disk, the DownloadModel branch emits
    NodeDownloadProgress + TaskStatusUpdated(Complete) and never starts a download."""
    import exo.worker.main as worker_main

    def _fake_resolve(*_args: Any, **_kwargs: Any) -> Any:
        return "/tmp/exo-model-dir"

    worker_main.resolve_existing_model = _fake_resolve  # type: ignore[assignment]
    worker_main.is_read_only_model_dir = lambda _p: False  # type: ignore[assignment]

    bound = _make_bound_runner()
    harness = _Harness(
        runner=_RecordingRunner(bound_instance=bound),
        state=State(
            instances={INSTANCE_1_ID: bound.instance},
            runners={RUNNER_1_ID: RunnerIdle()},
            downloads={NODE_A: []},
        ),
    )

    events = await harness.run_plan_once()

    assert harness.download_commands == []
    assert any(isinstance(ev, NodeDownloadProgress) for ev in events)
    completed = [ev for ev in events if isinstance(ev, TaskStatusUpdated)]
    assert any(ev.task_status == TaskStatus.Complete for ev in completed)


# ---------------------------------------------------------------------------
# plan_step: plain task forwarding to existing runner
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_plan_step_forwards_text_generation_to_runner() -> None:
    """A pending TextGeneration for a Ready runner forwards to start_task and
    emits TaskCreated."""
    bound = _make_single_node_bound_runner()
    runner = _RecordingRunner(bound_instance=bound, status=RunnerReady())
    text_task = _make_text_task()
    harness = _Harness(
        runner=runner,
        state=State(
            instances={INSTANCE_1_ID: bound.instance},
            runners={RUNNER_1_ID: RunnerReady()},
            tasks={TASK_1_ID: text_task},
        ),
    )

    events = await harness.run_plan_once()

    assert runner.started == [text_task]
    assert any(isinstance(ev, TaskCreated) for ev in events)


# ---------------------------------------------------------------------------
# _poll_connection_updates
# ---------------------------------------------------------------------------


def _make_net_state(
    *,
    self_node: NodeId,
    peer: NodeId,
    peer_ip: str,
    node_identities: dict[NodeId, NodeIdentity] | None = None,
) -> State:
    """Build a State with a self node and a peer node whose interface is known."""
    state = State(
        node_network={
            peer: NodeNetworkInfo(
                interfaces=[NetworkInterfaceInfo(name="en0", ip_address=peer_ip)]
            )
        },
        node_identities=node_identities if node_identities is not None else {},
    )
    state.topology.add_node(self_node)
    state.topology.add_node(peer)
    return state


async def _run_poll_once(
    harness: _Harness,
    *,
    reachable: list[Reachable] | None = None,
    fail: bool = False,
) -> list[Event]:
    """Drive exactly one _poll_connection_updates iteration with a stub
    check_reachable, canceling once the poll blocks for its next tick."""
    import exo.worker.main as worker_main

    results: list[Event] = []

    async def _collect() -> None:
        with harness.event_receiver:
            async for ev in harness.event_receiver:
                results.append(ev)

    async def _fake_poll(
        topology: Any,
        self_node_id: NodeId,
        node_network: Any,
        api_port: int,
        node_identities: Any = None,
    ) -> AsyncGenerator[Reachable, None]:
        if fail:
            return
        for item in reachable or []:
            yield item

    original = worker_main.check_reachable
    worker_main.check_reachable = _fake_poll  # type: ignore[assignment]

    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(_collect)
            await anyio.sleep(0)
            tg.start_soon(harness.worker._poll_connection_updates)
            await anyio.sleep(0.2)
            tg.cancel_scope.cancel()
    finally:
        worker_main.check_reachable = original

    await harness.event_sender.aclose()
    return results


@pytest.mark.anyio
async def test_poll_creates_edge_when_peer_discovered() -> None:
    """First discovery of a peer yields TopologyEdgeCreated with /ip4 multiaddr."""
    harness = _Harness()
    peer = NODE_B
    harness.worker.state = _make_net_state(self_node=NODE_A, peer=peer, peer_ip="10.0.0.5")

    events = await _run_poll_once(
        harness, reachable=[("10.0.0.5", peer, 12.5)]
    )

    created = [ev for ev in events if isinstance(ev, TopologyEdgeCreated)]
    assert len(created) == 1
    conn = created[0].conn
    assert conn.source == NODE_A
    assert conn.sink == peer
    assert isinstance(conn.edge, SocketConnection)
    assert conn.edge.sink_multiaddr.address == "/ip4/10.0.0.5/tcp/4001"
    assert conn.edge.latency_ms == 12.5


@pytest.mark.anyio
async def test_poll_replaces_edge_on_latency_change() -> None:
    """A material latency change deletes the old edge and creates a new one."""
    from exo.utils.info_gatherer.net_profile import latency_changed_materially

    peer = NODE_B
    harness = _Harness()
    harness.worker.state = _make_net_state(self_node=NODE_A, peer=peer, peer_ip="10.0.0.5")
    old_edge = SocketConnection(
        sink_multiaddr=Multiaddr(address="/ip4/10.0.0.5/tcp/4001"),
        latency_ms=10.0,
    )
    harness.worker.state.topology.add_connection(
        Connection(source=NODE_A, sink=peer, edge=old_edge)
    )
    new_latency = 100.0
    assert latency_changed_materially(10.0, new_latency)

    events = await _run_poll_once(
        harness, reachable=[("10.0.0.5", peer, new_latency)]
    )

    deleted = [ev for ev in events if isinstance(ev, TopologyEdgeDeleted)]
    created = [ev for ev in events if isinstance(ev, TopologyEdgeCreated)]
    assert len(deleted) == 1
    assert deleted[0].conn.edge == old_edge
    assert len(created) == 1
    assert created[0].conn.edge.latency_ms == new_latency


@pytest.mark.anyio
async def test_poll_does_not_churn_on_small_latency_jitter() -> None:
    """Sub-noise-floor latency jitter must NOT delete/recreate the edge."""
    peer = NODE_B
    harness = _Harness()
    harness.worker.state = _make_net_state(self_node=NODE_A, peer=peer, peer_ip="10.0.0.5")
    old_edge = SocketConnection(
        sink_multiaddr=Multiaddr(address="/ip4/10.0.0.5/tcp/4001"),
        latency_ms=10.0,
    )
    harness.worker.state.topology.add_connection(
        Connection(source=NODE_A, sink=peer, edge=old_edge)
    )

    events = await _run_poll_once(
        harness, reachable=[("10.0.0.5", peer, 10.4)]
    )

    assert not any(isinstance(ev, TopologyEdgeDeleted) for ev in events)
    assert not any(isinstance(ev, TopologyEdgeCreated) for ev in events)


@pytest.mark.anyio
async def test_poll_deletes_stale_edge_when_peer_unreachable() -> None:
    """A previously-known socket edge with no fresh probe result is deleted
    only after EDGE_DELETE_MISS_THRESHOLD consecutive missed polls — a single
    transient probe failure must NOT blink the peer out of the topology."""
    from exo.worker.main import EDGE_DELETE_MISS_THRESHOLD

    peer = NODE_B
    old_edge = SocketConnection(
        sink_multiaddr=Multiaddr(address="/ip4/10.0.0.5/tcp/4001"),
        latency_ms=10.0,
    )

    def _make_harness_with_edge() -> _Harness:
        harness = _Harness()
        harness.worker.state = _make_net_state(
            self_node=NODE_A, peer=peer, peer_ip="10.0.0.5"
        )
        harness.worker.state.topology.add_connection(
            Connection(source=NODE_A, sink=peer, edge=old_edge)
        )
        return harness

    # Each _run_poll_once closes its channels, so use a fresh harness whose
    # miss counter carries over from the previous iteration's worker state.
    harness = _make_harness_with_edge()
    for poll_no in range(1, EDGE_DELETE_MISS_THRESHOLD):
        events = await _run_poll_once(harness, reachable=[])
        deleted = [ev for ev in events if isinstance(ev, TopologyEdgeDeleted)]
        assert len(deleted) == 0, "edge must not be deleted on a transient miss"
        assert harness.worker._edge_probe_failures.get(old_edge) == poll_no
        # Carry the counter into the next iteration.
        harness = _make_harness_with_edge()
        harness.worker._edge_probe_failures[old_edge] = poll_no

    # Threshold-th consecutive miss: edge is finally deleted.
    events = await _run_poll_once(harness, reachable=[])
    deleted = [ev for ev in events if isinstance(ev, TopologyEdgeDeleted)]
    assert len(deleted) == 1
    assert deleted[0].conn.edge == old_edge
    assert old_edge not in harness.worker._edge_probe_failures


@pytest.mark.anyio
async def test_poll_resets_miss_counter_when_peer_rediscovered() -> None:
    """A miss followed by a rediscovery resets the counter — the edge must
    survive an arbitrary number of interleaved blips."""
    peer = NODE_B
    harness = _Harness()
    harness.worker.state = _make_net_state(self_node=NODE_A, peer=peer, peer_ip="10.0.0.5")
    old_edge = SocketConnection(
        sink_multiaddr=Multiaddr(address="/ip4/10.0.0.5/tcp/4001"),
        latency_ms=10.0,
    )
    harness.worker.state.topology.add_connection(
        Connection(source=NODE_A, sink=peer, edge=old_edge)
    )
    harness.worker._edge_probe_failures[old_edge] = 2  # two prior misses

    # Rediscovery: counter reset, no deletion.
    events = await _run_poll_once(harness, reachable=[("10.0.0.5", peer, 5.0)])
    deleted = [ev for ev in events if isinstance(ev, TopologyEdgeDeleted)]
    assert len(deleted) == 0
    assert harness.worker._edge_probe_failures.get(old_edge) is None


@pytest.mark.anyio
async def test_poll_uses_identity_api_port_for_multiaddr() -> None:
    """When the peer advertises a non-default api_port, the new edge uses it."""
    peer = NODE_B
    harness = _Harness(
        state=_make_net_state(
            self_node=NODE_A,
            peer=peer,
            peer_ip="10.0.0.5",
            node_identities={peer: NodeIdentity(api_port=4242)},
        )
    )

    events = await _run_poll_once(
        harness, reachable=[("10.0.0.5", peer, 5.0)]
    )

    created = [ev for ev in events if isinstance(ev, TopologyEdgeCreated)]
    assert len(created) == 1
    assert created[0].conn.edge.sink_multiaddr.port == 4242


@pytest.mark.anyio
async def test_poll_emits_nothing_when_probe_fails() -> None:
    """If check_reachable raises/returns nothing (e.g. transient network error),
    no topology events are emitted — the poll loop must survive."""
    harness = _Harness(
        state=_make_net_state(self_node=NODE_A, peer=NODE_B, peer_ip="10.0.0.5")
    )

    events = await _run_poll_once(harness, fail=True)

    assert events == []