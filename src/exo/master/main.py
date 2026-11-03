from datetime import datetime, timedelta, timezone

import anyio
from loguru import logger

from exo.master.placement import (
    add_instance_to_placements,
    cancel_unnecessary_downloads,
    delete_instance,
    get_transition_events,
    place_instance,
)
from exo.master.placement_utils import find_ip_prioritised
from exo.routing.event_router import (
    EventRouterBrokenResourceError,
    EventRouterClosedResourceError,
)
from exo.shared.apply import apply
from exo.shared.constants import EXO_EVENT_LOG_DIR, EXO_TRACING_ENABLED
from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.commands import (
    AddCustomModelCard,
    CreateInstance,
    DeleteCustomModelCard,
    DeleteInstance,
    DeleteInstanceLink,
    ForwarderCommand,
    ForwarderDownloadCommand,
    ImageEdits,
    ImageGeneration,
    PlaceInstance,
    PromoteMaster,
    RequestEventLog,
    SendInputChunk,
    SetInstanceLink,
    TaskCancelled,
    TaskFinished,
    TestCommand,
    TextGeneration,
)
from exo.shared.types.common import CommandId, NodeId, SessionId, SystemId
from exo.shared.types.events import (
    ChunkGenerated,
    CustomModelCardAdded,
    CustomModelCardDeleted,
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    InputChunkReceived,
    InstanceDeleted,
    InstanceLinkCreated,
    InstanceLinkDeleted,
    LocalForwarderEvent,
    NodeGatheredInfo,
    NodeTimedOut,
    TaskCreated,
    TaskDeleted,
    TaskStatusUpdated,
    TraceEventData,
    TracesCollected,
    TracesMerged,
)
from exo.shared.types.instance_link import InstanceLink
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    ImageEdits as ImageEditsTask,
)
from exo.shared.types.tasks import (
    ImageGeneration as ImageGenerationTask,
)
from exo.shared.types.tasks import (
    TaskId,
    TaskStatus,
)
from exo.shared.types.tasks import (
    TextGeneration as TextGenerationTask,
)
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runners import RunnerFailed
from exo.utils.channels import Receiver, Sender
from exo.utils.disk_event_log import DiskEventLog
from exo.utils.event_buffer import MultiSourceBuffer
from exo.utils.task_group import TaskGroup


def orphaned_download_node_ids(state: State) -> set[NodeId]:
    """Return download owners that are no longer members of the topology."""
    connected_node_ids = set(state.topology.list_nodes())
    return set(state.downloads).difference(connected_node_ids)


def _prefill_endpoint_for(state: State, decode_instance_id: InstanceId) -> str | None:
    decode = state.instances.get(decode_instance_id)
    if decode is None:
        return None
    decode_node = next(iter(decode.shard_assignments.node_to_runner.keys()), None)
    if decode_node is None:
        return None

    sources: set[InstanceId] = set()
    for link in state.instance_links.values():
        if decode_instance_id in link.decode_instances:
            sources.update(link.prefill_instances)
    sources.discard(decode_instance_id)

    in_flight = {TaskStatus.Pending, TaskStatus.Running}
    task_counts: dict[InstanceId, int] = {
        src_id: sum(
            1
            for task in state.tasks.values()
            if task.instance_id == src_id and task.task_status in in_flight
        )
        for src_id in sources
    }
    for src_id in sorted(sources, key=lambda sid: task_counts[sid]):
        instance = state.instances.get(src_id)
        if instance is None:
            continue
        for node_id, runner_id in instance.shard_assignments.node_to_runner.items():
            port = state.prefill_server_ports.get(runner_id)
            if port is None:
                continue
            ip = find_ip_prioritised(
                decode_node, node_id, state.topology, state.node_network, ring=True
            )
            if ip is None:
                continue
            return f"{ip}:{port}"
    return None


class Master:
    def __init__(
        self,
        node_id: NodeId,
        session_id: SessionId,
        *,
        command_receiver: Receiver[ForwarderCommand],
        event_sender: Sender[Event],
        local_event_receiver: Receiver[LocalForwarderEvent],
        global_event_sender: Sender[GlobalForwarderEvent],
        download_command_sender: Sender[ForwarderDownloadCommand],
    ):
        self.node_id = node_id
        self.session_id = session_id
        self.state = State()
        self._tg: TaskGroup = TaskGroup()
        self.command_task_mapping: dict[CommandId, TaskId] = {}
        # Tasks assigned this scheduling loop but not yet reflected in
        # self.state (TaskCreated still round-tripping through the event
        # router). Counted toward in-flight so a burst of concurrent requests
        # doesn't all read a stale zero and stampede onto one instance.
        self._pending_assignments: dict[TaskId, InstanceId] = {}
        self.command_receiver = command_receiver
        self.local_event_receiver = local_event_receiver
        self.global_event_sender = global_event_sender
        self.download_command_sender = download_command_sender
        self.event_sender = event_sender
        self._system_id = SystemId()
        self._multi_buffer = MultiSourceBuffer[SystemId, Event]()
        self._event_log = DiskEventLog(EXO_EVENT_LOG_DIR / "master")
        self._pending_traces: dict[TaskId, dict[int, list[TraceEventData]]] = {}
        while True:
            self._node_removed = anyio.Event()
            await self._remove_silent_nodes_and_broken_instances()
            with anyio.move_on_after(PLAN_INTERVAL):
                await self._node_removed.wait()

        while True:
            # Garbage-collect downloads left behind by node IDs that disappeared
            # before the master observed a last-seen timeout for them.
            # A node that is actively heartbeating (recent last_seen) but has
            # stale download entries is REJOINING, not retired — don't send
            # NodeTimedOut for it or it will be removed from topology again
            # the instant it re-announces (stable node IDs + one-shot
            # announcements make this a real race, seen live 08-07).
            now = datetime.now(tz=timezone.utc)
            for node_id in orphaned_download_node_ids(self.state):
                last = self.state.last_seen.get(node_id)
                if last is not None and now - last < node_inactivity_timeout:
                    continue
                logger.info(f"Removing downloads belonging to retired node {node_id}")
                await self.event_sender.send(NodeTimedOut(node_id=node_id))

            # kill broken instances
            connected_node_ids = set(self.state.topology.list_nodes())
            for instance_id, instance in self.state.instances.items():
                for node_id in instance.shard_assignments.node_to_runner:
                    if node_id not in connected_node_ids:
                        logger.warning(
                            f"Deleting instance {instance_id}: node {node_id} "
                            "is no longer in topology"
                        )
                        await self.event_sender.send(
                            InstanceDeleted(instance_id=instance_id)
                        )
                        break
                else:
                    # Keep instances with failed runners visible so users can
                    # read the error (RunnerFailed carries error_message +
                    # diagnostics and the dashboard renders it on FAILED
                    # cards). The runner process is already dead, so the
                    # instance is harmless — the user ejects it explicitly.
                    for rid in instance.shard_assignments.node_to_runner.values():
                        failed = self.state.runners.get(rid)
                        if isinstance(failed, RunnerFailed):
                            logger.warning(
                                f"Instance {instance_id} runner {rid} failed: "
                                f"{failed.error_message}"
                            )
                            break

            # time out dead nodes
            for node_id, time in self.state.last_seen.items():
                now = datetime.now(tz=timezone.utc)
                if now - time > node_inactivity_timeout:
                    logger.info(f"Manually removing node {node_id} due to inactivity")
                    await self.event_sender.send(NodeTimedOut(node_id=node_id))

    def _on_node_removed(self, node_id: NodeId) -> None:
        if node_id not in self._removing_nodes:
            # Not one this master timed out: the node announced it was leaving
            logger.info(f"Node {node_id} left the cluster")
        # Delete the instances it was part of now, not at the next check
        if self._node_removed is not None:
            self._node_removed.set()

    @staticmethod
    def _resend_due[K](sent: dict[K, float], key: K) -> bool:
        """Whether to send a removal for key: not sent yet, or sent a while ago and still
        not applied."""
        now = time.monotonic()
        if key in sent and now - sent[key] < REMOVAL_RESEND_INTERVAL:
            return False
        sent[key] = now
        return True

    async def _event_processor(self) -> None:
        with self.local_event_receiver as local_events:
            async for local_event in local_events:
                # Discard all events not from our session
                if local_event.session != self.session_id:
                    continue
                self._multi_buffer.ingest(
                    local_event.origin_idx,
                    local_event.event,
                    local_event.origin,
                )
                for event in self._multi_buffer.drain():
                    if isinstance(event, TracesCollected):
                        await self._handle_traces_collected(event)
                        continue

                    logger.debug(f"Master indexing event: {str(event)[:100]}")

                    event = event.model_copy(
                        update={"_master_time_stamp": datetime.now(tz=timezone.utc)}
                    )
                    if isinstance(event, NodeGatheredInfo):
                        event = event.model_copy(
                            update={"when": str(datetime.now(tz=timezone.utc))}
                        )

                    indexed = IndexedEvent(event=event, idx=len(self._event_log))
                    self.state = apply(self.state, indexed)

                    self._event_log.append(event)
                    await self._send_indexed_event(indexed)
                    if isinstance(event, NodeTimedOut):
                        self._on_node_removed(event.node_id)

    # This function is re-entrant, take care!
    async def _send_indexed_event(self, event: IndexedEvent):
        # Convenience method since this line is ugly
        await self.global_event_sender.send(
            GlobalForwarderEvent(
                origin=self.node_id,
                origin_idx=event.idx,
                session=self.session_id,
                event=event.event,
            )
        )

    async def _handle_traces_collected(self, event: TracesCollected) -> None:
        task_id = event.task_id
        if task_id not in self._pending_traces:
            self._pending_traces[task_id] = {}
        self._pending_traces[task_id][event.rank] = event.traces

        if (
            task_id in self._expected_ranks
            and set(self._pending_traces[task_id].keys())
            >= self._expected_ranks[task_id]
        ):
            await self._merge_and_save_traces(task_id)

    async def _merge_and_save_traces(self, task_id: TaskId) -> None:
        all_trace_data: list[TraceEventData] = []
        for trace_data in self._pending_traces[task_id].values():
            all_trace_data.extend(trace_data)

        await self.event_sender.send(
            TracesMerged(task_id=task_id, traces=all_trace_data)
        )

        del self._pending_traces[task_id]
        if task_id in self._expected_ranks:
            del self._expected_ranks[task_id]ENDOFFILE
