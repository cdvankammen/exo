"""Tests for the manual peer-join endpoint (POST /peers)."""
# pyright: reportUnusedFunction=false, reportAny=false
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.main import API


def _make_api(router: Any | None = None) -> Any:
    """Create a minimal API instance with the add-peer route."""
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api.router = router
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]
    app.post("/peers")(api.add_peer)
    return api


def _fake_httpx_get(text: str = "peer-node-id-123"):
    """Return a mock httpx.AsyncClient whose get() yields a response with `text`."""
    client = AsyncMock()
    resp = MagicMock()
    resp.text = text
    client.get = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def test_add_peer_connects() -> None:
    """POST /peers with a reachable host returns connected=True + node_id."""
    router = AsyncMock()
    router.connect_peer = AsyncMock(return_value=True)
    api = _make_api(router)
    client = TestClient(api.app)

    with patch("httpx.AsyncClient", return_value=_fake_httpx_get("peer-abc")):
        response = client.post(
            "/peers", json={"host": "mini2", "zenoh_port": 52414}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["host"] == "mini2"
    assert data["port"] == 52414
    assert data["connected"] is True
    assert data["node_id"] == "peer-abc"
    router.connect_peer.assert_awaited_once_with("mini2", 52414)


def test_add_peer_unreachable_returns_error() -> None:
    """If the peer's HTTP API is unreachable, connected=False + error."""
    router = AsyncMock()
    api = _make_api(router)
    client = TestClient(api.app)

    client_mock = AsyncMock()
    client_mock.get = AsyncMock(side_effect=ConnectionError("connection refused"))
    client_mock.__aenter__ = AsyncMock(return_value=client_mock)
    client_mock.__aexit__ = AsyncMock(return_value=False)
    with patch("httpx.AsyncClient", return_value=client_mock):
        response = client.post("/peers", json={"host": "10.0.0.99"})

    assert response.status_code == 200
    data = response.json()
    assert data["connected"] is False
    assert "not reachable" in (data["error"] or "")
    # zenoh dial must NOT happen if the API is unreachable
    router.connect_peer.assert_not_called()


def test_add_peer_default_port() -> None:
    """zenoh_port defaults to 52414 when omitted."""
    router = AsyncMock()
    router.connect_peer = AsyncMock(return_value=True)
    api = _make_api(router)
    client = TestClient(api.app)

    with patch("httpx.AsyncClient", return_value=_fake_httpx_get("peer-abc")):
        response = client.post("/peers", json={"host": "tailscale-name"})

    assert response.status_code == 200
    data = response.json()
    assert data["port"] == 52414
    assert data["connected"] is True
    router.connect_peer.assert_awaited_once_with("tailscale-name", 52414)


def test_add_peer_no_router_returns_503() -> None:
    """Without a router wired, the endpoint returns 503."""
    api = _make_api(router=None)
    client = TestClient(api.app)

    response = client.post("/peers", json={"host": "mini2"})

    assert response.status_code == 503


def test_add_peer_bad_host_returns_400() -> None:
    """A zenoh dial failure surfaces as a 400 with the error detail."""
    router = AsyncMock()
    router.connect_peer = AsyncMock(side_effect=RuntimeError("resolve failed"))
    api = _make_api(router)
    client = TestClient(api.app)

    with patch("httpx.AsyncClient", return_value=_fake_httpx_get("peer-abc")):
        response = client.post("/peers", json={"host": "no-such-host"})

    assert response.status_code == 400
    assert "resolve failed" in response.text