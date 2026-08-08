# pyright: reportUnusedFunction=false, reportAny=false
from typing import Any
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.shared.types.commands import PromoteMaster
from exo.shared.types.common import NodeId


def _make_api() -> Any:
    """Create a minimal API instance with promote_master route + error handler."""
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api._send = AsyncMock()  # pyright: ignore[reportPrivateUsage]

    class _Topology:
        def list_nodes(self) -> list[str]:
            return []

    class _State:
        topology = _Topology()
        instances: dict[str, object] = {}

    api.state = _State()  # pyright: ignore[reportAttributeAccessIssue]
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]
    app.post("/master/promote/{node_id}")(api.promote_master)
    return api


def test_promote_master_unknown_node_404() -> None:
    """Promoting a node not in the topology returns 404 in OpenAI error format."""
    api = _make_api()
    client = TestClient(api.app)

    response = client.post("/master/promote/unknown-node")
    assert response.status_code == 404
    data: dict[str, Any] = response.json()
    assert "error" in data
    assert data["error"]["message"] == "Node not found"
    assert data["error"]["code"] == 404


def test_promote_master_with_running_instances_409() -> None:
    """Promoting while instances run returns 409 (idle-only promotion)."""
    api = _make_api()
    api.state.instances = {"inst-1": object()}
    api.state.topology.list_nodes = lambda: ["node-a", "node-b"]
    client = TestClient(api.app)

    response = client.post("/master/promote/node-a")
    assert response.status_code == 409
    data: dict[str, Any] = response.json()
    assert "error" in data
    assert "instances are running" in data["error"]["message"]
    api._send.assert_not_called()


def test_promote_master_happy_path() -> None:
    """Promoting a known idle node sends the PromoteMaster command + returns 200."""
    api = _make_api()
    api.state.topology.list_nodes = lambda: ["node-a", "node-b"]
    client = TestClient(api.app)

    response = client.post("/master/promote/node-a")
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data["message"] == "Command received."
    assert data["target_node_id"] == "node-a"
    api._send.assert_called_once()
    command = api._send.call_args[0][0]
    assert isinstance(command, PromoteMaster)
    assert command.target_node_id == NodeId("node-a")
