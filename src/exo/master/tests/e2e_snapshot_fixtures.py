"""Fixtures shared by the state-snapshot E2E tests.

Real Master + real EventRouter + real apply(); the master->node hop goes
through the real STATE_SNAPSHOTS topic serialization.

"""

from datetime import datetime, timedelta, timezone

from loguru import logger

from exo.master.main import REPLAYABLE_EVENTS
from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    Event,
    NodeGatheredInfo,
    TestEvent,
)
from exo.utils.info_gatherer.info_gatherer import MiscData

logger.remove()
N = REPLAYABLE_EVENTS + 25


def event_for(i: int, node: NodeId) -> Event:
    # `when` must be an ISO timestamp: apply() parses it.
    when = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=i)
    if i % 2:
        return NodeGatheredInfo(
            node_id=node,
            when=when.isoformat(),
            info=MiscData(friendly_name=f"node-{i}"),
        )
    return TestEvent()
