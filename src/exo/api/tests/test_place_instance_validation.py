# pyright: reportUnusedFunction=false, reportAny=false
"""POST /place_instance refuses a placement the cluster can't hold, instead of accepting it."""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.api.types.api import PlaceInstanceParams
from exo.master.tests.conftest import create_node_memory, create_node_network
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask, card_cache
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import ForwarderCommand, PlaceInstance
from exo.shared.types.common import NodeId, SystemId
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.utils.channels import Receiver, channel

MODEL = ModelCard(
    model_id=ModelId("test-org/test-model"),
    storage_size=Memory.from_kb(1000),
    n_layers=10,
    hidden_size=30,
    supports_tensor=True,
    tasks=[ModelTask.TextGeneration],
    backends=[Backend.MlxMetal],
)
NODE = NodeId("only-node")


@pytest.fixture(autouse=True)
def known_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(card_cache, "cc", {MODEL.model_id: MODEL})


def single_node_api() -> tuple[API, Receiver[ForwarderCommand]]:
    topology = Topology()
    topology.add_node(NODE)
    api = object.__new__(API)
    api.state = State(
        topology=topology,
        node_memory={NODE: create_node_memory(10_000_000)},
        node_network={NODE: create_node_network()},
        node_backends={NODE: [Backend.MlxMetal]},
    )
    api.paused = False
    api._system_id = SystemId()  # pyright: ignore[reportPrivateUsage]
    api.command_sender, commands = channel[ForwarderCommand]()
    return api, commands


def _topology_with(*node_ids: NodeId) -> Topology:
    """A topology containing the given nodes (add_node wires no edges)."""
    topology = Topology()
    for node_id in node_ids:
        topology.add_node(node_id)
    return topology


async def test_a_placement_that_fits_is_sent_to_the_master() -> None:
    api, commands = single_node_api()

    await api.place_instance(PlaceInstanceParams(model_id=MODEL.model_id))

    assert [type(c.command) for c in commands.collect()] == [PlaceInstance]


async def test_a_placement_that_cannot_fit_is_refused_with_the_reason() -> None:
    api, commands = single_node_api()

    with pytest.raises(HTTPException) as refused:
        await api.place_instance(
            PlaceInstanceParams(model_id=MODEL.model_id, min_nodes=2)
        )

    assert refused.value.status_code == 400
    assert refused.value.detail
    assert commands.collect() == []


async def test_the_refusal_reaches_the_caller_as_400_over_http() -> None:
    """The endpoint answers 400 with the reason, not 200 "ok"."""
    api, commands = single_node_api()
    app = FastAPI()
    api.app = app
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]
    app.post("/place_instance")(api.place_instance)

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/place_instance", json={"model_id": str(MODEL.model_id), "min_nodes": 2}
    )

    assert response.status_code == 400
    assert commands.collect() == []


async def test_force_override_still_reaches_the_master() -> None:
    """force_override bypasses the memory check, so it must not be refused here.

    The knob the master honours has to reach the pre-check too, or a deliberate
    "load it anyway" would be refused by a check that didn't know about it.
    """
    api, commands = single_node_api()
    # 1 MB of storage against a node reporting only 1 KB available: refused
    # without the override, accepted with it.
    api.state = api.state.model_copy(
        update={"node_memory": {NODE: create_node_memory(1_000)}}
    )

    await api.place_instance(
        PlaceInstanceParams(model_id=MODEL.model_id, force_override=True)
    )

    assert [type(c.command) for c in commands.collect()] == [PlaceInstance]


async def test_oversized_model_is_refused_without_force_override() -> None:
    """Not enough memory anywhere: refused with the reason, nothing sent."""
    api, commands = single_node_api()
    # 1 MB of storage against a node reporting only 1 KB available.
    api.state = api.state.model_copy(
        update={"node_memory": {NODE: create_node_memory(1_000)}}
    )

    with pytest.raises(HTTPException) as refused:
        await api.place_instance(PlaceInstanceParams(model_id=MODEL.model_id))

    assert refused.value.status_code == 400
    assert "memory" in str(refused.value.detail).lower()
    assert commands.collect() == []


async def test_memory_tolerance_is_accepted_and_forwarded() -> None:
    """memory_tolerance is plumbed through to the command, not validated against.

    It is threaded from the request into both the pre-check and the command
    (and exposed on GET /instance/placement), but no placement code reads it
    yet, so it cannot change the outcome. Assert the request is accepted and
    the value survives onto the command, rather than asserting a placement
    difference the knob does not make.
    """
    api, commands = single_node_api()

    await api.place_instance(
        PlaceInstanceParams(model_id=MODEL.model_id, memory_tolerance=0.5)
    )

    sent = commands.collect()
    assert len(sent) == 1
    assert isinstance(sent[0].command, PlaceInstance)
    assert sent[0].command.memory_tolerance == 0.5


async def test_node_ids_is_accepted_and_forwarded() -> None:
    """node_ids reaches the command, but no placement code reads it yet.

    Like memory_tolerance, it is plumbed from the request into both the pre-check
    and the command and is exposed on GET /instance/placement, but place_instance
    never reads command.node_ids, so it cannot change the outcome. Assert the
    value survives onto the command rather than a placement difference it does
    not make.
    """
    api, commands = single_node_api()
    absent = NodeId("no-such-node")

    await api.place_instance(
        PlaceInstanceParams(model_id=MODEL.model_id, node_ids=[absent])
    )

    sent = commands.collect()
    assert len(sent) == 1
    assert isinstance(sent[0].command, PlaceInstance)
    assert sent[0].command.node_ids == [absent]


async def test_node_layers_reaches_the_pre_check() -> None:
    """A manual layer allocation over an unconnected node pair is refused.

    node_layers is validated inside the same placement call the master makes, so
    the refusal (here: "No connected cycle exactly matches the manual layer
    allocation nodes") proves the pre-check is given the caller's real request.
    """
    api, commands = single_node_api()
    other = NodeId("some-other-node")
    api.state = api.state.model_copy(update={"topology": _topology_with(NODE, other)})

    with pytest.raises(HTTPException) as refused:
        await api.place_instance(
            PlaceInstanceParams(
                model_id=MODEL.model_id,
                node_layers={NODE: 5, other: 5},
            )
        )

    assert refused.value.status_code == 400
    assert commands.collect() == []
