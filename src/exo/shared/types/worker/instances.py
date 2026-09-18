"""Instance types.

:class:`InstanceId`/:class:`InstanceMeta` and the concrete MLX ring/JACCL instances plus :class:`BoundInstance`."""

from enum import Enum

from pydantic import model_validator

from exo.shared.models.model_cards import ModelTask
from exo.shared.types.common import Host, Id, NodeId
from exo.shared.types.worker.runners import RunnerId, ShardAssignments, ShardMetadata
from exo.utils.pydantic_ext import FrozenModel, TaggedModel


class InstanceId(Id):
    """Identifier of a model instance."""


class InstanceMeta(str, Enum):
    """Instance flavour (backend implementation) selector."""

    MlxRing = "MlxRing"
    MlxJaccl = "MlxJaccl"


class BaseInstance(TaggedModel):
    """Base class for concrete instances: id + shard assignments."""

    instance_id: InstanceId
    shard_assignments: ShardAssignments

    def shard(self, runner_id: RunnerId) -> ShardMetadata | None:
        """Return the shard assigned to ``runner_id``, or None."""
        return self.shard_assignments.runner_to_shard.get(runner_id, None)


class MlxRingInstance(BaseInstance):
    """MLX ring instance: per-node host lists plus an ephemeral port."""

    hosts_by_node: dict[NodeId, list[Host]]
    ephemeral_port: int


class MlxJacclInstance(BaseInstance):
    """MLX JACCL instance: RDMA device map per rank pair + coordinators."""

    # jaccl_devices[i][j] lists the RDMA interface names on rank i that
    # connect to rank j, one entry per physical link. Entry k of [i][j] and
    # entry k of [j][i] are the two ends of the same link.
    jaccl_devices: list[list[list[str]]]
    jaccl_coordinators: dict[NodeId, str]


# TODO: Single node instance
Instance = MlxRingInstance | MlxJacclInstance


class BoundInstance(FrozenModel):
    """An instance bound to a specific runner on a specific node."""

    instance: Instance
    bound_runner_id: RunnerId
    bound_node_id: NodeId

    @property
    def bound_shard(self) -> ShardMetadata:
        """Return the shard assigned to the bound runner."""
        shard = self.instance.shard(self.bound_runner_id)
        assert shard is not None
        return shard

    @property
    def is_image_model(self) -> bool:
        """Whether the bound model supports image tasks."""
        return (
            ModelTask.TextToImage in self.bound_shard.model_card.tasks
            or ModelTask.ImageToImage in self.bound_shard.model_card.tasks
        )

    @model_validator(mode="after")
    def validate_shard_exists(self) -> "BoundInstance":
        """Require the bound runner to own a shard in the instance."""
        assert (
            self.bound_runner_id in self.instance.shard_assignments.runner_to_shard
        ), (
            "Bound Instance must be constructed with a runner_id that is in the instances assigned shards"
        )
        return self
