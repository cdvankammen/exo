# pyright: reportUnusedFunction=false, reportAny=false
"""Tests for multi-tenant API key auth + per-key quota enforcement.

Covers ``exo.api.tenant`` directly (middleware factory + registry), plus the
API wiring when tenant sources exist. Mirrors the existing ``test_api_auth.py``
toy-app pattern so nothing here needs a live cluster.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from exo.api.tenant import (
    TenantDB,
    TenantKeySpec,
    TenantRegistry,
    meter_chat_endpoint_middleware,
    multi_tenant_auth_middleware,
)

KEY_A = "tenant-key-aaa"
KEY_B = "tenant-key-bbb"


def _make_app(
    registry: TenantRegistry,
    *,
    legacy_token: str | None = None,
    meter: bool = False,
) -> Any:
    app = FastAPI()

    @app.get("/v1/models")
    def _models() -> dict[str, str]:
        return {"ok": "true"}

    @app.post("/v1/chat/completions")
    async def _chat(request: Request) -> dict[str, object]:
        # Re-read the cached body (the meter stored it) to prove streaming
        # is not consumed by the middleware.
        body = getattr(request.state, "_body", None)
        return {"ok": "true", "body_read": body is not None}

    # NOTE: FastAPI applies http middleware in REVERSE registration order —
    # the LAST registered runs FIRST (outermost). The meter must run AFTER
    # auth establishes request.state.tenant, so register meter FIRST.
    if meter:
        app.middleware("http")(meter_chat_endpoint_middleware(registry))
    app.middleware("http")(
        multi_tenant_auth_middleware(registry, legacy_token=legacy_token)
    )
    return app


@pytest.fixture()
def registry(tmp_path) -> TenantRegistry:
    db = TenantDB(path=tmp_path / "tenant.db")
    reg = TenantRegistry(static={}, db=db)
    reg.create_key(
        key=KEY_A,
        rate_limit=5,
        daily_tokens=1000,
        models=("*",),
        display_name="tenant A",
    )
    reg.create_key(
        key=KEY_B,
        rate_limit=2,
        daily_tokens=100,
        models=("qwen-3.8-max",),
        display_name="tenant B",
    )
    return reg


def _client(reg: TenantRegistry, **kw) -> TestClient:
    return TestClient(_make_app(reg, **kw))


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_unknown_key_401(registry: TenantRegistry) -> None:
    client = _client(registry)
    resp = client.get("/v1/models", headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 401
    assert resp.json()["error"]["error_code"] == "UNAUTHORIZED"


def test_missing_header_401(registry: TenantRegistry) -> None:
    client = _client(registry)
    resp = client.get("/v1/models")
    assert resp.status_code == 401


def test_valid_key_200(registry: TenantRegistry) -> None:
    client = _client(registry)
    resp = client.get(
        "/v1/models", headers={"Authorization": f"Bearer {KEY_A}"}
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": "true"}


def test_legacy_token_wildcard(registry: TenantRegistry) -> None:
    client = _client(registry, legacy_token="admin-legacy-token")
    resp = client.get(
        "/v1/models", headers={"Authorization": "Bearer admin-legacy-token"}
    )
    assert resp.status_code == 200
    # Admin wildcard bypasses quota metering entirely.
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "qwen-3.8-max", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer admin-legacy-token"},
    )
    assert resp.status_code == 200
    usage = registry.usage("admin-legacy-token")
    assert usage.rate_count(1e18) == 0  # not tracked


def test_dashboard_paths_exempt(registry: TenantRegistry) -> None:
    client = _client(registry)
    # The middleware keeps is_public_dashboard_path logic as defense-in-depth.
    from exo.api.auth import is_public_dashboard_path

    assert is_public_dashboard_path("/")
    assert is_public_dashboard_path("/_app/x.js")
    assert not is_public_dashboard_path("/v1/models")


# ---------------------------------------------------------------------------
# Rate limits
# ---------------------------------------------------------------------------


def test_rate_limit_429_with_retry_after(registry: TenantRegistry) -> None:
    client = _client(registry)
    # KEY_A allows 5 per 60s.
    for _ in range(5):
        resp = client.get("/v1/models", headers={"Authorization": f"Bearer {KEY_A}"})
        assert resp.status_code == 200
    resp = client.get("/v1/models", headers={"Authorization": f"Bearer {KEY_A}"})
    assert resp.status_code == 429
    body = resp.json()
    assert body["error"]["error_code"] == "RATE_LIMITED"
    assert "Retry-After" in resp.headers


def test_rate_limit_per_key_isolated(registry: TenantRegistry) -> None:
    client = _client(registry)
    # KEY_B allows only 2; KEY_A allows 5.
    for _ in range(2):
        resp = client.get("/v1/models", headers={"Authorization": f"Bearer {KEY_B}"})
        assert resp.status_code == 200
    assert (
        client.get("/v1/models", headers={"Authorization": f"Bearer {KEY_B}"}).status_code
        == 429
    )
    # KEY_A is unaffected.
    resp = client.get("/v1/models", headers={"Authorization": f"Bearer {KEY_A}"})
    assert resp.status_code == 200


def test_rejected_requests_not_recorded(registry: TenantRegistry) -> None:
    """A burst of rejects must not push the key further into rate-limit debt."""
    client = _client(registry)
    for _ in range(5):
        client.get("/v1/models", headers={"Authorization": f"Bearer {KEY_A}"})
    # Now in rate-limit.  Rejected calls must NOT extend the window.
    for _ in range(3):
        client.get("/v1/models", headers={"Authorization": f"Bearer {KEY_A}"})
    usage = registry.usage(KEY_A)
    assert usage.rate_count(1e18) == 5  # only the accepted 5 recorded


# ---------------------------------------------------------------------------
# Token quota (metering)
# ---------------------------------------------------------------------------


def test_token_quota_exceeded_429(registry: TenantRegistry) -> None:
    client = _client(registry, meter=True)
    # KEY_B: daily_tokens=100 → ~40 tokens per "hi" request (4 chars/token
    # estimate: content "hi" = 2 chars → 1 token? No: max(1, 2//4)=1 token).
    # Use a long message to exceed 100 tokens in one shot.
    big_content = "x" * 500  # 500 chars // 4 = 125 tokens > 100
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen-3.8-max",
            "messages": [{"role": "user", "content": big_content}],
        },
        headers={"Authorization": f"Bearer {KEY_B}"},
    )
    assert resp.status_code == 429
    body = resp.json()
    assert body["error"]["error_code"] == "QUOTA_EXCEEDED"


def test_token_quota_within_cap_passes(registry: TenantRegistry) -> None:
    client = _client(registry, meter=True)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen-3.8-max",
            "messages": [{"role": "user", "content": "hello"}],
        },
        headers={"Authorization": f"Bearer {KEY_B}"},
    )
    assert resp.status_code == 200
    # Body was cached so the route could re-read it without stream-consumption.
    assert resp.json()["body_read"] is True


def test_non_metered_endpoint_no_quota(registry: TenantRegistry) -> None:
    """GET /v1/models doesn't consume token quota (only rate)."""
    client = _client(registry, meter=True)
    # KEY_A allows 5 requests/60s; all 5 succeed and none consume tokens.
    for _ in range(5):
        resp = client.get("/v1/models", headers={"Authorization": f"Bearer {KEY_A}"})
        assert resp.status_code == 200
    usage = registry.usage(KEY_A)
    # Every recorded stamp must have est_tokens == 0 for non-metered routes.
    assert all(est == 0 for _, est in usage.request_stamps)
    assert usage.day_total("2099-01-01") == 0


# ---------------------------------------------------------------------------
# DB key store
# ---------------------------------------------------------------------------


def test_db_upsert_and_delete(tmp_path) -> None:
    db = TenantDB(path=tmp_path / "tenant.db")
    db.upsert("k1", rate_limit=10, daily_tokens=100)
    spec = db.get("k1")
    assert spec is not None
    assert spec.rate_limit == 10
    assert spec.daily_tokens == 100
    assert db.delete("k1") is True
    assert db.get("k1") is None
    assert db.delete("k1") is False  # idempotent


def test_db_wins_over_static(tmp_path) -> None:
    db = TenantDB(path=tmp_path / "tenant.db")
    db.upsert("shared", rate_limit=99, daily_tokens=999)
    reg = TenantRegistry(
        static={"shared": TenantKeySpec("shared", 1, 1)}, db=db
    )
    spec = reg.resolve("shared")
    assert spec is not None
    assert spec.rate_limit == 99  # DB wins


def test_static_spec_from_json(tmp_path, monkeypatch) -> None:
    cfg_file = tmp_path / "tenants.json"
    cfg_file.write_text(
        json.dumps(
            {
                "static-key": {
                    "rate_limit": 7,
                    "daily_tokens": 500,
                    "models": ["model-a", "model-b"],
                    "note": "bootstrapped",
                }
            }
        )
    )
    monkeypatch.setattr("exo.api.tenant.TENANT_JSON_PATH", cfg_file)
    from exo.api.tenant import _load_static_specs

    reg = TenantRegistry(static=_load_static_specs(), db=TenantDB(path=tmp_path / "t2.db"))
    spec = reg.resolve("static-key")
    assert spec is not None
    assert spec.rate_limit == 7
    assert spec.models == ("model-a", "model-b")


def test_model_allowlist_enforced(registry: TenantRegistry) -> None:
    """is_allowed_model reflects per-key model allowlists."""
    assert registry.is_allowed_model(KEY_A, "anything")  # wildcard
    assert registry.is_allowed_model(KEY_B, "qwen-3.8-max")
    assert not registry.is_allowed_model(KEY_B, "other-model")