# pyright: reportUnusedFunction=false, reportAny=false
"""Tests for GET /v1/logs/{node_id}/{name} — per-node log tail proxy.

Covers the local-node shortcut, remote proxying through the node identity
api_host/api_port, unknown-node 404s, unreachable-node 502s, remote log-not-found
404s, and the lines/tail semantics passthrough.
"""
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api import main as api_main
from exo.api.main import API


def _make_api() -> Any:
    """Create a minimal API instance with the per-node log route registered."""
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]
    api.node_id = "node-local-1"  # type: ignore[attr-defined]
    api.state = type("S", (), {"node_identities": {}})()  # type: ignore[attr-defined]
    app.get("/v1/logs/{node_id}/{name}")(api.get_log_node)
    return api


@pytest.fixture(autouse=True)
def _isolated_log_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the whitelist at a temp dir so tests never touch a real ~/.exo log."""
    log_path = tmp_path / "exo.log"
    monkeypatch.setattr(api_main, "_LOG_FILES", {"main": log_path})
    return log_path


class _Identity:
    api_host = "10.0.0.9"
    api_port = 8000
    friendly_name = "mini2"


class _FakeAsyncClient:
    """Stand-in for httpx.AsyncClient that returns a canned payload."""

    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self.payload = payload
        self.status = status

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *a: object) -> None:
        return None

    async def get(self, url: str, timeout: float | None = None) -> Any:
        class _Resp:
            def __init__(self, payload: dict[str, Any], status: int) -> None:
                self._payload = payload
                self.status_code = status

            def raise_for_status(self) -> None:
                if self.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        "error", request=None, response=self  # pyright: ignore[reportArgumentType]
                    )

            def json(self) -> dict[str, Any]:
                return self._payload

        return _Resp(self.payload, self.status)


def _patch_httpx(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any], status: int = 200):
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: _FakeAsyncClient(payload=payload, status=status)
    )


# ── Local node shortcut ────────────────────────────────────────────────────


def test_get_log_node_local_shortcut(_isolated_log_files: Path) -> None:
    """Requesting our own node_id serves the local tail without HTTP."""
    _isolated_log_files.write_text("line0\nline1\nline2\n")
    api = _make_api()
    client = TestClient(api.app)

    response = client.get("/v1/logs/node-local-1/main?lines=2")
    assert response.status_code == 200
    data = response.json()
    assert data["nodeId"] == "node-local-1"
    assert data["nodeName"] == "node-loc"  # unknown node → short id fallback
    assert data["name"] == "main"
    assert data["content"] == "line1\nline2"
    assert data["truncated"] is True


def test_get_log_node_local_shortcut_unknown_log(_isolated_log_files: Path) -> None:
    api = _make_api()
    client = TestClient(api.app)

    response = client.get("/v1/logs/node-local-1/does_not_exist")
    assert response.status_code == 404


# ── Remote proxying ────────────────────────────────────────────────────────


def test_get_log_node_remote_proxy(_isolated_log_files: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A known peer with a valid api endpoint is proxied and enriched."""
    api = _make_api()
    api.state = type("S", (), {"node_identities": {"node-remote-9": _Identity()}})()  # type: ignore[attr-defined]

    _patch_httpx(
        monkeypatch,
        {"name": "main", "content": "remote line", "truncated": False},
    )

    client = TestClient(api.app)
    response = client.get("/v1/logs/node-remote-9/main?lines=5")
    assert response.status_code == 200
    data = response.json()
    assert data["nodeId"] == "node-remote-9"
    assert data["nodeName"] == "mini2"
    assert data["name"] == "main"
    assert data["content"] == "remote line"
    assert data["truncated"] is False


def test_get_log_node_unknown_node_404(_isolated_log_files: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A node that is not in state.node_identities returns 404 before any HTTP."""
    api = _make_api()
    api.state = type("S", (), {"node_identities": {"node-remote-9": _Identity()}})()  # type: ignore[attr-defined]
    _patch_httpx(monkeypatch, {})

    client = TestClient(api.app)
    response = client.get("/v1/logs/node-missing/main")
    assert response.status_code == 404


def test_get_log_node_node_without_api_404(_isolated_log_files: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A node that does not advertise api_host/api_port returns 404."""
    class _NoApi:
        api_host = ""
        api_port = 0

    api = _make_api()
    api.state = type("S", (), {"node_identities": {"node-dark": _NoApi()}})()  # type: ignore[attr-defined]
    _patch_httpx(monkeypatch, {})

    client = TestClient(api.app)
    response = client.get("/v1/logs/node-dark/main")
    assert response.status_code == 404


def test_get_log_node_remote_404_propagates(_isolated_log_files: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 404 from the target node maps to LogNotFound, not a bare proxy error."""
    api = _make_api()
    api.state = type("S", (), {"node_identities": {"node-remote-9": _Identity()}})()  # type: ignore[attr-defined]
    _patch_httpx(monkeypatch, {"detail": "not found"}, status=404)

    client = TestClient(api.app)
    response = client.get("/v1/logs/node-remote-9/main")
    assert response.status_code == 404


def test_get_log_node_remote_error_502(_isolated_log_files: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 500 from the target node maps to a proxy error."""
    api = _make_api()
    api.state = type("S", (), {"node_identities": {"node-remote-9": _Identity()}})()  # type: ignore[attr-defined]
    _patch_httpx(monkeypatch, {}, status=500)

    client = TestClient(api.app)
    response = client.get("/v1/logs/node-remote-9/main")
    assert response.status_code == 502


def test_get_log_node_unreachable_502(_isolated_log_files: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A connection failure to the target node maps to 502 NodeUnreachable."""
    api = _make_api()
    api.state = type("S", (), {"node_identities": {"node-remote-9": _Identity()}})()  # type: ignore[attr-defined]

    async def _boom(url: str, timeout: float | None = None) -> Any:
        raise ConnectionError("refused")

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: type("C", (), {"__aenter__": lambda s: s, "__aexit__": lambda *a: None, "get": _boom})(),
    )

    client = TestClient(api.app)
    response = client.get("/v1/logs/node-remote-9/main")
    assert response.status_code == 502