from datetime import datetime
from typing import final

from pydantic import Field

from exo.shared.models.model_cards import ModelCard
from exo.shared.topology import Connection
from exo.shared.types.chunks import Chunk, InputImageChunk
from exo.shared.types.common import CommandId, Id, ModelId, NodeId, SessionId, SystemId
from exo.shared.types.instance_link import InstanceLink, InstanceLinkId
from exo.shared.types.tasks import Task, TaskId, TaskStatus
from exo.shared.types.worker.downloads import DownloadProgress
from exo.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from exo.shared.types.worker.runners import RunnerId, RunnerStatus
from exo.shared.types.worker.shards import Sharding
from exo.utils.info_gatherer.info_gatherer import GatheredInfo
from exo.utils.pydantic_ext import FrozenModel, TaggedModel


class EventId(Id):
    """
    Newtype around `ID`
    """


class BaseEvent(TaggedModel):
    event_id: EventId = Field(default_factory=EventId)
    # Internal, for debugging. Please don't rely on this field for anything!
    _master_time_stamp: None | datetime = None


class TestEvent(BaseEvent):
    __test__ = False


class TaskCreated(BaseEvent):
    task_id: TaskId
    task: Task


class TaskAcknowledged(BaseEvent):
    task_id: TaskId


class TaskDeleted(BaseEvent):
    task_id: TaskId


class TaskStatusUpdated(BaseEvent):
    task_id: TaskId
    task_status: TaskStatus


class TaskFailed(BaseEvent):
    task_id: TaskId
    error_type: str
    error_message: str


class InstanceCreated(BaseEvent):
    instance: Instance

    def __eq__(self, other: object) -> bool:
        if isinstance(other, InstanceCreated):
            return self.instance == other.instance and self.event_id == other.event_id

        return False


class InstanceDeleted(BaseEvent):
    instance_id: InstanceId


class RunnerStatusUpdated(BaseEvent):
    runner_id: RunnerId
    runner_status: RunnerStatus


class NodeTimedOut(BaseEvent):
    node_id: NodeId


# TODO: bikeshed this name
class NodeGatheredInfo(BaseEvent):
    node_id: NodeId
    when: str  # this is a manually cast datetime overrode by the master when the event is indexed, rather than the local time on the device
    info: GatheredInfo


class NodeDownloadProgress(BaseEvent):
    download_progress: DownloadProgress


class ChunkGenerated(BaseEvent):
    command_id: CommandId
    chunk: Chunk


class InputChunkReceived(BaseEvent):
    command_id: CommandId
    chunk: InputImageChunk


class TopologyEdgeCreated(BaseEvent):
    conn: Connection


class TopologyEdgeDeleted(BaseEvent):
    conn: Connection


class CustomModelCardAdded(BaseEvent):
    model_card: ModelCard


class CustomModelCardDeleted(BaseEvent):
    model_id: ModelId


@final
class TraceEventData(FrozenModel):
    name: str
    start_us: int
    duration_us: int
    rank: int
    category: str


@final
class TracesCollected(BaseEvent):
    task_id: TaskId
    rank: int
    traces: list[TraceEventData]


@final
class TracesMerged(BaseEvent):
    task_id: TaskId
    traces: list[TraceEventData]


class InstanceLinkCreated(BaseEvent):
    link: InstanceLink


class InstanceLinkDeleted(BaseEvent):
    link_id: InstanceLinkId


class PrefixIndexEvent(BaseEvent):
    """Cluster-wide prefix-cache index event (advisory, compact).

    Published by a runner after it caches a prompt prefix (add/update of
    the KV prefix cache), and absorbed by the worker-side
    ``ClusterPrefixIndex`` (see ``exo.worker.engines.mlx.cluster_cache``).
    Carries only hashes + lengths + identity — never KV tensors.
    """

    model_hash: str
    chunks: tuple[str, ...]
    token_count: int
    node_id: str
    instance_id: str
    last_used: float = Field(default_factory=lambda: 0.0)
    hits: int = 0


class MasterHeartbeat(BaseEvent):
    """Periodic liveness signal emitted by the master.

    Carries no payload — its presence in the stream is the signal. Treated
    as a pass-through event by ``exo.shared.apply`` (does not mutate state).
    """


class PlacementForcedOverride(BaseEvent):
    """Audit record emitted whenever a placement used ``force_override=True``.

    Force override deliberately bypasses placement guard rails, so every use
    is recorded with enough context to answer *who*, *what*, and *which
    check was skipped*. The event itself is a pass-through (does not mutate
    :class:`~exo.shared.types.state.State`); it is indexed into the master
    event log so it can be replayed and searched, and forwarded to every node
    on ``GLOBAL_EVENTS`` so cluster-wide observers see the override.
    """

    command_id: CommandId
    model_id: ModelId
    sharding: Sharding
    instance_meta: InstanceMeta
    min_nodes: int
    node_ids: list[str] | None = None
    node_layers: dict[str, int] | None = None
    memory_tolerance: float = 1.0
    # Which specific checks were bypassed because force_override=True.
    bypassed_guardrails: list[str]
    # Who originated the request. The API sets this from the incoming HTTP
    # request (dashboard session / API user); commands already in flight
    # carry the originating SystemId.
    source: str = "unknown"
    # True when the override actually changed the outcome (e.g. a placement
    # that would otherwise have been rejected). False when force_override was
    # set but nothing was skipped in practice.
    effective: bool = True


Event = (
    TestEvent
    | TaskCreated
    | TaskStatusUpdated
    | TaskFailed
    | TaskDeleted
    | TaskAcknowledged
    | InstanceCreated
    | InstanceDeleted
    | RunnerStatusUpdated
    | NodeTimedOut
    | NodeGatheredInfo
    | NodeDownloadProgress
    | ChunkGenerated
    | InputChunkReceived
    | TopologyEdgeCreated
    | TopologyEdgeDeleted
    | TracesCollected
    | TracesMerged
    | CustomModelCardAdded
    | CustomModelCardDeleted
    | InstanceLinkCreated
    | InstanceLinkDeleted
    | PrefixIndexEvent
    | MasterHeartbeat
    | PlacementForcedOverride
)


class IndexedEvent(FrozenModel):
    """An event indexed by the master, with a globally unique index"""

    idx: int = Field(ge=0)
    event: Event


class GlobalForwarderEvent(FrozenModel):
    """An event the forwarder will serialize and send over the network"""

    origin_idx: int = Field(ge=0)
    origin: NodeId
    session: SessionId
    event: Event


class LocalForwarderEvent(FrozenModel):
    """An event the forwarder will serialize and send over the network"""

    origin_idx: int = Field(ge=0)
    origin: SystemId
    session: SessionId
    event: Event
