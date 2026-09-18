# pyright: reportUnusedFunction=false, reportAny=false
from typing import Any
from unittest.mock import AsyncMock

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from exo.api.main import API

DEPRECATED_CHAT_PATH = "/ollama/api/api/chat"
DEPRECATED_TAGS_PATH = "/ollama/api/api/tags"
CANONICAL_CHAT_PATH = "/ollama/api/chat"
CANONICAL_TAGS_PATH = "/ollama/api/tags"
DEPRECATION_HEADER = "X-EXO-Deprecation"
DISPOSITION_HEADER = "Deprecation"


def _make_api() -> Any:
    """Build a minimal API with a real FastAPI app."""
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api.state = None
    api._send = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]
    return api


def test_ollama_deprecated_alias_routes_are_registered() -> None:
    """The typo-alias routes must still exist during the deprecation window."""
    api = _make_api()
    api._setup_routes()  # pyright: ignore[reportPrivateUsage]
    paths = {getattr(r, "path", None) for r in api.app.routes}
    assert DEPRECATED_CHAT_PATH in paths
    assert DEPRECATED_TAGS_PATH in paths


def test_ollama_deprecated_alias_returns_deprecation_headers() -> None:
    """Typo-alias responses must carry an explicit deprecation signal."""
    api = _make_api()
    response = {"models": []}
    api.ollama_tags = AsyncMock(return_value=response)  # pyright: ignore[reportPrivateUsage]
    wrapped = api._ollama_deprecated_alias(api.ollama_tags)  # pyright: ignore[reportPrivateUsage]
    api.app.get(DEPRECATED_TAGS_PATH)(wrapped)

    client = TestClient(api.app)

    resp = client.get(DEPRECATED_TAGS_PATH)
    assert resp.status_code == 200
    assert resp.headers.get(DEPRECATION_HEADER) == "true"
    assert DISPOSITION_HEADER in resp.headers


def test_ollama_canonical_routes_have_no_deprecation_headers() -> None:
    """Canonical routes must not be marked deprecated."""
    api = _make_api()

    async def fake_tags(request: Request) -> dict[str, Any]:
        return {"models": []}

    api.app.get(CANONICAL_TAGS_PATH)(fake_tags)

    client = TestClient(api.app)

    resp = client.get(CANONICAL_TAGS_PATH)
    assert resp.status_code == 200
    assert DEPRECATION_HEADER not in resp.headers
    assert DISPOSITION_HEADER not in resp.headers


def test_ollama_deprecated_alias_chat_route_preserves_payload() -> None:
    """POST to the deprecated chat alias must still reach the real handler."""
    api = _make_api()
    api.ollama_chat = AsyncMock(return_value={"message": "ok"})  # pyright: ignore[reportPrivateUsage]
    wrapped = api._ollama_deprecated_alias(api.ollama_chat)  # pyright: ignore[reportPrivateUsage]
    api.app.post(DEPRECATED_CHAT_PATH)(wrapped)

    client = TestClient(api.app)

    resp = client.post(DEPRECATED_CHAT_PATH, json={"model": "x", "messages": []})
    assert resp.status_code == 200
    assert resp.json() == {"message": "ok"}
    assert resp.headers.get(DEPRECATION_HEADER) == "true"


def test_ollama_deprecated_alias_forwards_args() -> None:
    """The wrapper must forward request/args/kwargs to the inner handler."""
    api = _make_api()
    seen = {}

    async def fake_tags(request, *args, **kwargs):
        seen["path"] = request.url.path
        return {"models": []}

    wrapped = api._ollama_deprecated_alias(fake_tags)  # pyright: ignore[reportPrivateUsage]
    api.app.get(DEPRECATED_TAGS_PATH)(wrapped)

    client = TestClient(api.app)
    resp = client.get(DEPRECATED_TAGS_PATH)
    assert resp.status_code == 200
    assert seen["path"] == DEPRECATED_TAGS_PATH
    assert DISPOSITION_HEADER in resp.headers