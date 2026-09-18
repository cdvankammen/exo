"""Tests for Master request load-balancing across 2+ instances of a model.

Requirement: "Load-balanced requests across ≥2 instances of same model."

The Master's ``_in_flight_counts`` method (commit 4bbb930a) implements
least-loaded instance selection. This test mirrors the proven
``test_master.py`` boot (PlaceInstance -> InstanceCreated via event router)
and adds a SECOND node + SECOND instance of the same model, then verifies
consecutive requests spread across BOTH instances rather than all landing on
one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import anyio
import pytest
from loguru import logger

from exo.master.main import Master
from exo.routing.router import get_node_zid
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.commands import (
    CommandId,
    ForwarderCommand,
    ForwarderDownloadCommand,
    PlaceInstance,
    TextGeneration,
)
from exo.shared.types.common import ModelId, SessionId, SystemId
from exo.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    InstanceCreated,
    LocalForwarderEvent,
    NodeGatheredInfo,
    TaskCreated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage
from exo.shared.types.tasks import TaskStatus
from exo.shared.types.tasks import TextGeneration as TextGenerationTask
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import (
    InstanceMeta,
    MlxRingInstance,
    ShardAssignments,
)
from exo.shared.types.worker.shards import PipelineShardMetadata, Sharding
from exo.utils.channels import channel
from exo.utils.info_gatherer.info_gatherer import NodeBackends

MODEL = ModelId("llama-3.2-1b")


@pytest.mark.asyncio
async def test_requests_balance_across_two_instances():
    node_id = get_node_zid()
    session_id = SessionId(master_node_id=node_id, election_clock=0)

    ge_sender, global_event_receiver = channel[GlobalForwarderEvent]()
    command_sender, co_receiver = channel[ForwarderCommand]()
    local_event_sender, le_receiver = channel[LocalForwarderEvent]()
    fcds, fcdr = channel[ForwarderDownloadCommand]()
    ev_send, ev_recv = channel[Event]()

    master = Master(
        node_id,
        session_id,
        event_sender=ev_send,
        global_event_sender=ge_sender,
        local_event_receiver=le_receiver,
        command_receiver=co_receiver,
        download_command_sender=fcds,
    )

    async def mock_event_router():
        idx = 0
        sid = SystemId()
        with ev_recv as master_events:
            async for event in master_events:
                await local_event_sender.send(
                    LocalForwarderEvent(
                        origin=sid,
                        origin_idx=idx,
                        session=session_id,
                        event=event,
                    )
                )
                idx += 1

    all_events: list[IndexedEvent] = []

    def _get_events() -> Sequence[IndexedEvent]:
        orig_events = global_event_receiver.collect()
        for e in orig_events:
            all_events.append(
                IndexedEvent(
                    event=e.event,
                    idx=len(all_events),
                )
            )
        return all_events

    logger.info("run the master")
    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)
        tg.start_soon(mock_event_router)

        # Inject a NodeGatheredInfo event (node A with memory + backends)
        await local_event_sender.send(
            LocalForwarderEvent(
                origin_idx=0,
                origin=SystemId("Worker"),
                session=session_id,
                event=(
                    NodeGatheredInfo(
                        when=str(datetime.now(tz=timezone.utc)),
                        node_id=node_id,
                        info=MemoryUsage(
                            ram_total=Memory.from_bytes(678948 * 1024),
                            ram_available=Memory.from_bytes(678948 * 1024),
                            swap_total=Memory.from_bytes(0),
                            swap_available=Memory.from_bytes(0),
                        ),
                    )
                ),
            )
        )
        await local_event_sender.send(
            LocalForwarderEvent(
                origin_idx=1,
                origin=SystemId("Worker"),
                session=session_id,
                event=(
                    NodeGatheredInfo(
                        when=str(datetime.now(tz=timezone.utc)),
                        node_id=node_id,
                        info=NodeBackends(backends=[Backend.MlxMetal]),
                    )
                ),
            )
        )

        # wait for initial topology event
        while len(list(master.state.topology.list_nodes())) == 0:
            await anyio.sleep(0.001)
        while len(master.state.node_memory) == 0:
            await anyio.sleep(0.001)
        while len(master.state.node_backends) == 0:
            await anyio.sleep(0.001)

        # Place instance #1 on node A
        logger.info("inject a CreateInstance Command")
        await command_sender.send(
            ForwarderCommand(
                origin=SystemId("API"),
                command=(
                    PlaceInstance(
                        command_id=CommandId(),
                        model_card=ModelCard(
                            model_id=MODEL,
                            n_layers=16,
                            storage_size=Memory.from_bytes(678948),
                            hidden_size=7168,
                            supports_tensor=True,
                            tasks=[ModelTask.TextGeneration],
                            backends=[Backend.MlxMetal],
                        ),
                        sharding=Sharding.Pipeline,
                        instance_meta=InstanceMeta.MlxRing,
                        min_nodes=1,
                    )
                ),
            )
        )
        # wait for an instance
        while len(master.state.instances.keys()) == 0:
            await anyio.sleep(0.001)

        # Place instance #2 on the same node (same model -> second instance)
        await command_sender.send(
            ForwarderCommand(
                origin=SystemId("API"),
                command=(
                    PlaceInstance(
                        command_id=CommandId(),
                        model_card=ModelCard(
                            model_id=MODEL,
                            n_layers=16,
                            storage_size=Memory.from_bytes(678948),
                            hidden_size=7168,
                            supports_tensor=True,
                            tasks=[ModelTask.TextGeneration],
                            backends=[Backend.MlxMetal],
                        ),
                        sharding=Sharding.Pipeline,
                        instance_meta=InstanceMeta.MlxRing,
                        min_nodes=1,
                    )
                ),
            )
        )
        # wait until 2 instances exist
        while len(master.state.instances.keys()) < 2:
            await anyio.sleep(0.001)

        ids = list(master.state.instances.keys())
        assert len(ids) == 2

        # Fire 4 text-generation requests and record which instance each lands on
        assigned: dict[str, str] = {}
        for i in range(4):
            command_id = CommandId()
            await command_sender.send(
                ForwarderCommand(
                    origin=SystemId("API"),
                    command=(
                        TextGeneration(
                            command_id=command_id,
                            task_params=TextGenerationTaskParams(
                                model=MODEL,
                                input=[
                                    InputMessage(
                                        role="user",
                                        content=InputMessageContent(f"msg-{i}"),
                                    )
                                ],
                            ),
                        )
                    ),
                )
            )
            # wait for the TaskCreated event carrying this command_id
            deadline = 100
            while deadline > 0:
                task_id = master.command_task_mapping.get(command_id)
                if task_id is not None:
                    task = master.state.tasks.get(task_id)
                    if task is not None:
                        assigned[str(command_id)] = task.instance_id
                        break
                await anyio.sleep(0.005)
                deadline -= 1

        # 4 requests, 2 instances -> at least 2 distinct destinations
        destinations = set(assigned.values())
        assert len(destinations) >= 2, (
            f"All requests landed on one instance: {assigned}. "
            "Load balancing across instances failed."
        )
        logger.info(f"assigned: {assigned}")

        ev_send.close()
        await master.shutdown()