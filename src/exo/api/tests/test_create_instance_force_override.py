# pyright: reportUnusedFunction=false, reportAny=false
from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import ShardAssignments


def _make_api() -> Any:
    """Create a minimal API instance with the create-instance route."""
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api.state = _make_low_memory_state()
    api._send = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]
    app.post("/instance")(api.create_instance)
    return api


def _make_low_memory_state() -> State:
    """A State with a single node that has only 100 bytes available."""
    node_id = NodeId()
    return State(
        node_memory={
            node_id: MemoryUsage.from_bytes(
                ram_total=1000, ram_available=100, swap_total=0, swap_available=0
            )
        }
    )


def _make_instance() -> MlxRingInstance:
    """A minimal single-node ring instance for the test-model."""
    return MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=ModelId("test-model"),
            runner_to_shard={},
            node_to_runner={},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )


def _make_oversized_card() -> ModelCard:
    """A model card whose storage size exceeds the node's available memory."""
    return ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_kb(1001),
        n_layers=10,
        hidden_size=1000,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


def test_create_instance_blocks_without_override() -> None:
    """Without force_override, an oversized model is rejected with 400."""
    api = _make_api()
    card = _make_oversized_card()

    with patch("exo.api.main.ModelCard.load", new=AsyncMock(return_value=card)):
        client = TestClient(api.app)
        response = client.post(
            "/instance", json={"instance": _make_instance().model_dump()}
        )

    assert response.status_code == 400
    assert "Insufficient memory" in response.text
    api._send.assert_not_called()


def test_create_instance_allows_with_override() -> None:
    """With force_override=True, an oversized model is accepted and the
    CreateInstance command is sent."""
    api = _make_api()
    card = _make_oversized_card()
    instance = _make_instance()

    with patch("exo.api.main.ModelCard.load", new=AsyncMock(return_value=card)):
        client = TestClient(api.app)
        response = client.post(
            "/instance",
            json={
                "instance": instance.model_dump(),
                "force_override": True,
            },
        )

    assert response.status_code == 200
    api._send.assert_called_once()
    command = api._send.call_args[0][0]
    assert command.instance.instance_id == instance.instance_id