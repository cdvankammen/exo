"""Command envelope types from master to nodes.

:class:`BaseCommand` and its subclasses for text/image generation, instance placement, downloads, instance links, and master promotion."""

from pydantic import Field

from exo.api.types import (
    ImageEditsTaskParams,
    ImageGenerationTaskParams,
)
from exo.shared.models.model_cards import ModelCard, ModelId
from exo.shared.types.chunks import InputImageChunk
from exo.shared.types.common import CommandId, NodeId, SystemId
from exo.shared.types.instance_link import InstanceLinkId
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from exo.shared.types.worker.shards import Sharding, ShardMetadata
from exo.utils.pydantic_ext import FrozenModel, TaggedModel


class BaseCommand(TaggedModel):
    """Base class for all commands sent from master to nodes (tagged)."""

    command_id: CommandId = Field(default_factory=CommandId)


class TestCommand(BaseCommand):
    """Test command; excluded from pytest collection via ``__test__``."""

    __test__ = False


class TextGeneration(BaseCommand):
    """Command to generate text with the given task params."""

    task_params: TextGenerationTaskParams


class ImageGeneration(BaseCommand):
    """Command to generate images with the given task params."""

    task_params: ImageGenerationTaskParams


class ImageEdits(BaseCommand):
    """Command to edit an image with the given task params."""

    task_params: ImageEditsTaskParams


class PlaceInstance(BaseCommand):
    """Command to place (materialize) a model instance across nodes.

    Carries the model card, sharding plan, optional explicit node subset,
    and memory-constraint knobs (``force_override``, ``memory_tolerance``).
    """

    model_card: ModelCard
    sharding: Sharding
    instance_meta: InstanceMeta
    min_nodes: int
    node_layers: dict[NodeId, int] | None = None
    # Optional explicit node subset for placement. When set, only cycles that
    # exactly match these nodes are considered (same semantics as the
    # node_ids query param on the preview/compatibility endpoints).
    node_ids: set[NodeId] | None = None
    # When True, skip memory-sufficiency checks (cycles/layers). Intended for
    # users who explicitly want to attempt loading a model that exceeds the
    # reported available memory ("load the model anyway").
    force_override: bool = False
    # Memory tolerance multiplier (0.0–1.0). 1.0 = strict (model must fit
    # available memory). Lower values relax the check: e.g. 0.5 admits a
    # cycle holding at least half the model's size. force_override=True
    # bypasses the check entirely regardless of this value.
    memory_tolerance: float = 1.0


class CreateInstance(BaseCommand):
    """Command to create an already-formed instance directly."""

    instance: Instance


class DeleteInstance(BaseCommand):
    """Command to delete an instance by id."""

    instance_id: InstanceId


class TaskCancelled(BaseCommand):
    """Command to cancel a previously-issued command."""

    cancelled_command_id: CommandId


class TaskFinished(BaseCommand):
    """Command signalling completion of a previously-issued command."""

    finished_command_id: CommandId


class SendInputChunk(BaseCommand):
    """Command to send an input image chunk (converted to event by master)."""

    chunk: InputImageChunk


class RequestEventLog(BaseCommand):
    """Command requesting the event log from the given index onwards."""

    since_idx: int


class StartDownload(BaseCommand):
    """Command to start downloading a model shard onto a target node.

    ``force_override`` bypasses the disk-space sufficiency check.
    """

    target_node_id: NodeId
    shard_metadata: ShardMetadata
    # When True, bypass the disk-space sufficiency check and attempt to
    # download the model anyway ("download the model anyway").
    force_override: bool = False


class DeleteDownload(BaseCommand):
    """Command to delete a downloaded model shard from a node."""

    target_node_id: NodeId
    model_id: ModelId


class CancelDownload(BaseCommand):
    """Command to cancel an in-flight model shard download."""

    target_node_id: NodeId
    model_id: ModelId


class AddCustomModelCard(BaseCommand):
    """Command to register a custom model card."""

    model_card: ModelCard


class DeleteCustomModelCard(BaseCommand):
    """Command to remove a custom model card by model id."""

    model_id: ModelId


class SetInstanceLink(BaseCommand):
    """Command to (re)wire prefill/decode instances for an instance link."""

    link_id: InstanceLinkId
    prefill_instances: list[InstanceId]
    decode_instances: list[InstanceId]


class DeleteInstanceLink(BaseCommand):
    """Command to remove an instance link by id."""

    link_id: InstanceLinkId


class PromoteMaster(BaseCommand):
    """Force target_node_id to win the next master election.

    Delivered to every node (topics.COMMANDS is broadcast) -- only the node
    whose own id matches target_node_id acts on it.
    """

    target_node_id: NodeId


DownloadCommand = StartDownload | DeleteDownload | CancelDownload


Command = (
    TestCommand
    | RequestEventLog
    | TextGeneration
    | ImageGeneration
    | ImageEdits
    | PlaceInstance
    | CreateInstance
    | DeleteInstance
    | TaskCancelled
    | TaskFinished
    | SendInputChunk
    | AddCustomModelCard
    | DeleteCustomModelCard
    | SetInstanceLink
    | DeleteInstanceLink
    | PromoteMaster
)


class ForwarderCommand(FrozenModel):
    """Envelope wrapping a command with its originating system id."""

    origin: SystemId
    command: Command


class ForwarderDownloadCommand(FrozenModel):
    """Envelope wrapping a download command with its originating system id."""

    origin: SystemId
    command: DownloadCommand
