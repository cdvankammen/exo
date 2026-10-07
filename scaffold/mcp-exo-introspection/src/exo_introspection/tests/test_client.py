"""Client + end-to-end tool tests with httpx MockTransport (no cluster)."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from exo_introspection.client import ExoClient
from exo_introspection.errors import (
    ExoAuthError,
    ExoConnectionError,
    ExoNotFoundError,
    ExoPlacementError,
)


def _mock_client(handler, base_url="http://test", api_token=None) -> ExoClient:
    client = ExoClient(base_url=base_url, api_token=api_token)
    client._client = httpx.AsyncClient(
        base_url=base_url, transport=httpx.MockTransport(handler)
    )
    return client


def _cleanup(client: ExoClient) -> None:
    asyncio.run(client.aclose())


# --------------------------------------------------------------------------- #
# client adapter
# --------------------------------------------------------------------------- #

def test_client_get_state_parses_json(state_single):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/state"
        return httpx.Response(200, json=state_single)

    client = _mock_client(handler)
    payload = asyncio.run(client.get_state())
    assert payload["topology"]["nodes"] == state_single["topology"]["nodes"]
    assert "downloads" in payload
    _cleanup(client)


def test_client_sends_bearer_token():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"topology": {"nodes": [], "connections": {}}})

    client = _mock_client(handler, api_token="sekrit")
    asyncio.run(client.get_state())
    assert captured["auth"] == "Bearer sekrit"
    _cleanup(client)


def test_client_auth_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"detail": "nope"})

    client = _mock_client(handler, api_token="bad")
    with pytest.raises(ExoAuthError):
        asyncio.run(client.get_state())
    _cleanup(client)


def test_client_placement_error_maps_code(placement_failure_body):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json=placement_failure_body)

    client = _mock_client(handler)
    with pytest.raises(ExoPlacementError) as excinfo:
        asyncio.run(client.get_placement(model_id="m"))
    assert excinfo.value.extra["error_code"] == "PLACEMENT_FAILED"
    _cleanup(client)


def test_client_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Unknown model"})

    client = _mock_client(handler)
    with pytest.raises(ExoNotFoundError):
        asyncio.run(client.get_state())
    _cleanup(client)


# --------------------------------------------------------------------------- #
# end-to-end: tool handler -> client -> mock exo API
# --------------------------------------------------------------------------- #

def _app_with_client(client: ExoClient):
    """Build an app whose closures use the given mock-backed client.

    Mirrors server.py wiring exactly (kept in sync with the real server;
    this bypasses create_server's own client construction so tests don't
    need a live cluster).
    """
    from mcp.server.fastmcp import FastMCP

    from exo_introspection.errors import ExoError, ExoPlacementError
    from exo_introspection.explain import (
        build_reason,
        summarize_placement,
        summarize_placement_failure,
    )
    from exo_introspection.normalize import (
        normalize_downloads,
        normalize_nodes,
        normalize_topology,
    )

    app = FastMCP("exo-introspection")

    @app.tool()
    async def list_nodes(include_state: bool = False) -> dict:
        state = await client.get_state()
        return {
            "ok": True,
            "result": {
                "nodes": normalize_nodes(state, include_state),
                "count": len((state.get("topology") or {}).get("nodes", [])),
            },
        }

    @app.tool()
    async def get_topology(
        include_cycles: bool = False, include_node_details: bool = False
    ) -> dict:
        state = await client.get_state()
        return {"ok": True, "result": normalize_topology(state, include_cycles)}

    @app.tool()
    async def explain_placement(
        model_id: str,
        sharding: str = "Pipeline",
        instance_meta: str = "MlxRing",
        min_nodes: int = 1,
        memory_tolerance: float = 1.0,
        force_override: bool = False,
    ) -> dict:
        params = {
            "model_id": model_id,
            "sharding": sharding,
            "instance_meta": instance_meta,
            "min_nodes": min_nodes,
            "memory_tolerance": memory_tolerance,
            "force_override": force_override,
        }
        try:
            placement = await client.get_placement(**params)
        except ExoPlacementError as exc:
            previews = None
            try:
                previews = await client.get_placement_previews(**params)
            except ExoError:
                previews = None
            return {
                "ok": True,
                "result": summarize_placement_failure(
                    model_id, exc.to_payload(), previews
                ),
            }
        except ExoError as exc:
            return {"ok": False, "error": exc.to_payload()}
        summary = summarize_placement(placement)
        if not summary.get("reason"):
            summary["reason"] = build_reason(placement, sharding, min_nodes)
        return {"ok": True, "result": summary}

    @app.tool()
    async def get_download_status(
        model_id: str | None = None, status: str | None = None
    ) -> dict:
        state = await client.get_state()
        rows = normalize_downloads(state, model_id=model_id, status=status)
        summary = {"by_status": {}}
        for row in rows:
            summary["by_status"][row["status"]] = (
                summary["by_status"].get(row["status"], 0) + 1
            )
        return {"ok": True, "result": {"downloads": rows, "count": len(rows), "summary": summary}}

    return app


def test_tool_list_nodes_e2e(state_multi):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=state_multi)

    client = _mock_client(handler)
    app = _app_with_client(client)
    tools = app._tool_manager._tools
    assert set(tools) == {"list_nodes", "get_topology", "explain_placement", "get_download_status"}

    result = asyncio.run(tools["list_nodes"].fn(include_state=True))
    assert result["ok"] is True
    assert result["result"]["count"] == 2

    result = asyncio.run(tools["get_topology"].fn(include_cycles=True))
    assert result["ok"] is True
    assert len(result["result"]["edges"]) == 3
    assert result["result"]["rdma_cycles_available"] is True
    _cleanup(client)


def test_tool_explain_placement_success(placement_success):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/instance/placement"
        assert request.url.params["model_id"] == "mlx-community/Llama-3.2-3B-4bit"
        assert request.url.params["sharding"] == "Pipeline"
        return httpx.Response(200, json=placement_success)

    client = _mock_client(handler)
    app = _app_with_client(client)
    tools = app._tool_manager._tools

    result = asyncio.run(
        tools["explain_placement"].fn(model_id="mlx-community/Llama-3.2-3B-4bit")
    )
    assert result["ok"] is True
    assert result["result"]["sharding"] == "Pipeline"
    assert result["result"]["score_factors"]["accelerator_score"] == 2
    assert len(result["result"]["shard_assignments"]) == 2
    _cleanup(client)


def test_tool_explain_placement_failure_falls_back_to_previews(
    placement_failure_body, placement_previews_body
):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/instance/placement":
            return httpx.Response(400, json=placement_failure_body)
        if request.url.path == "/instance/previews":
            return httpx.Response(200, json=placement_previews_body)
        return httpx.Response(404, json={})

    client = _mock_client(handler)
    app = _app_with_client(client)
    tools = app._tool_manager._tools

    result = asyncio.run(
        tools["explain_placement"].fn(model_id="mlx-community/Llama-3.2-3B-4bit")
    )
    assert calls == ["/instance/placement", "/instance/previews"]
    # placement failure is a *rich result*, not a hard error
    assert result["ok"] is True
    assert result["result"]["ok"] is False
    assert result["result"]["error"]["code"] == "PLACEMENT_FAILED"
    assert len(result["result"]["candidates"]) == 2
    _cleanup(client)


def test_tool_download_status_e2e(state_multi):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=state_multi)

    client = _mock_client(handler)
    app = _app_with_client(client)
    tools = app._tool_manager._tools

    result = asyncio.run(tools["get_download_status"].fn())
    assert result["ok"] is True
    assert result["result"]["count"] == 5
    assert result["result"]["summary"]["by_status"]["completed"] == 1

    result = asyncio.run(
        tools["get_download_status"].fn(model_id="mlx-community/Llama-3.2-3B-4bit", status="failed")
    )
    assert result["ok"] is True
    assert result["result"]["count"] == 1
    assert result["result"]["downloads"][0]["error_message"] == "checksum mismatch"
    _cleanup(client)


# --------------------------------------------------------------------------- #
# required base URL — no baked-in default port
# --------------------------------------------------------------------------- #

def test_missing_base_url_raises(monkeypatch):
    """No EXO_MCP_BASE_URL and no explicit arg → refuse to guess a port."""
    monkeypatch.delenv("EXO_MCP_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="EXO_MCP_BASE_URL"):
        ExoClient()


def test_empty_env_base_url_raises(monkeypatch):
    """An empty EXO_MCP_BASE_URL (as shipped in config) → same refusal."""
    monkeypatch.setenv("EXO_MCP_BASE_URL", "")
    with pytest.raises(ValueError, match="EXO_MCP_BASE_URL"):
        ExoClient()


def test_env_base_url_is_used(monkeypatch):
    monkeypatch.setenv("EXO_MCP_BASE_URL", "http://env-host:1234")
    client = ExoClient()
    assert client.base_url == "http://env-host:1234"
    _cleanup(client)


def test_explicit_base_url_beats_env(monkeypatch):
    monkeypatch.setenv("EXO_MCP_BASE_URL", "http://env-host:1234")
    client = ExoClient(base_url="http://explicit:9999/")
    assert client.base_url == "http://explicit:9999"
    _cleanup(client)