# pyright: reportUnusedFunction=false, reportAny=false
"""Tests for the opt-in API bearer-token auth (T23).

Covers ``exo.api.auth`` (the middleware factory) directly. Existing API
tests bypass ``API.__init__`` where the middleware is installed, so they
are unaffected when EXO_API_TOKEN is unset.
"""

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.auth import api_token_auth_middleware, is_public_dashboard_path

TOKEN = "test-secret-token"


def _make_app(token: str | None) -> Any:
    app = FastAPI()

    @app.get("/v1/models")
    def _models() -> dict[str, str]:
        return {"ok": "true"}

    @app.get("/instance/abc")
    def _instance() -> dict[str, str]:
        return {"id": "abc"}

    if token is not None:
        app.middleware("http")(api_token_auth_middleware(token))
    return app


def test_no_token_configured_allows_all() -> None:
    client = TestClient(_make_app(None))
    assert client.get("/v1/models").status_code == 200
    assert client.get("/instance/abc").status_code == 200


def test_missing_header_401() -> None:
    client = TestClient(_make_app(TOKEN))
    resp = client.get("/v1/models")
    assert resp.status_code == 401
    assert resp.json()["error"]["error_code"] == "UNAUTHORIZED"


def test_wrong_token_401() -> None:
    client = TestClient(_make_app(TOKEN))
    resp = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_correct_token_200() -> None:
    client = TestClient(_make_app(TOKEN))
    resp = client.get("/v1/models", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 200


def test_dashboard_paths_exempt() -> None:
    client = TestClient(_make_app(TOKEN))
    # Public paths bypass auth entirely.
    assert is_public_dashboard_path("/")
    assert is_public_dashboard_path("/_app/immutable/chunk.js")
    assert is_public_dashboard_path("/favicon.ico")
    assert not is_public_dashboard_path("/v1/models")
    assert not is_public_dashboard_path("/instance/abc")
    # The static mount itself is not registered here, so a 404 (not 401) is fine.
    assert client.get("/").status_code in (200, 404)
    assert client.get("/_app/x.js").status_code in (200, 404)
