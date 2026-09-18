"""Instance-link state types.

:class:`InstanceLinkId` and :class:`InstanceLink` model inter-instance communication chains in the topology."""

from exo.shared.types.common import Id
from exo.shared.types.worker.instances import InstanceId
from exo.utils.pydantic_ext import FrozenModel


class InstanceLinkId(Id):
    pass


class InstanceLink(FrozenModel):
    link_id: InstanceLinkId
    prefill_instances: list[InstanceId]
    decode_instances: list[InstanceId]
