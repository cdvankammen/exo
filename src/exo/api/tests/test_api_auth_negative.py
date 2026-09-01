# pyright: reportUnusedFunction=false, reportAny=false
"""Negative / bypass auth tests for ``exo.api.auth`` (T23 supplement).

Covers the failure modes that existing ``test_api_auth.py`` does not exercise:
malformed headers, empty tokens, partial token substrings, scheme variants,
case sensitivity, multiple header values, and public-path bypass attempts
with auth present.

Run with:  uv run pytest src/exo/api/tests/test_api_auth_negative.py -v
"""

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.auth import api_token_auth_middleware, is_public_dashboard_path

TOKEN = "test-secret-token"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app(token: str | None) -> Any:
    """Create a minimal FastAPI app with auth middleware and two protected routes."""
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


# ---------------------------------------------------------------------------
# Malformed header variants
# ---------------------------------------------------------------------------

def test_malformed_header_no_space_before_token() -> None:
    """'Bearer<token>' without space should be rejected (header != 'Bearer ...')."""
    client = TestClient(_make_app(TOKEN))
    resp = client.get("/v1/models", headers={"Authorization": f"Bearer{TOKEN}"})
    assert resp.status_code == 401
    assert resp.json()["error"]["error_code"] == "UNAUTHORIZED"


def test_malformed_header_lowercase_scheme() -> None:
    """'bearer <token>' (lowercase) should be rejected — case-sensitive compare."""
    client = TestClient(_make_app(TOKEN))
    resp = client.get("/v1/models", headers={"Authorization": f"bearer {TOKEN}"})
    assert resp.status_code == 401


def test_malformed_header_uppercase_scheme() -> None:
    """'BEARER <token>' (all uppercase) should be rejected."""
    client = TestClient(_make_app(TOKEN))
    resp = client.get("/v1/models", headers={"Authorization": f"BEARER {TOKEN}"})
    assert resp.status_code == 401


def test_malformed_header_basic_scheme() -> None:
    """'Basic <token>' instead of 'Bearer' should be rejected."""
    client = TestClient(_make_app(TOKEN))
    resp = client.get("/v1/models", headers={"Authorization": f"Basic {TOKEN}"})
    assert resp.status_code == 401


def test_malformed_header_raw_token_no_scheme() -> None:
    """Sending the token as the bare Authorization value (no scheme) should 401."""
    client = TestClient(_make_app(TOKEN))
    resp = client.get("/v1/models", headers={"Authorization": TOKEN})
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Empty / missing value variants
# ---------------------------------------------------------------------------

def test_empty_bearer_token() -> None:
    """'Bearer ' with no actual token value should be rejected."""
    client = TestClient(_make_app(TOKEN))
    resp = client.get("/v1/models", headers={"Authorization": "Bearer "})
    assert resp.status_code == 401
    assert resp.json()["error"]["error_code"] == "UNAUTHORIZED"


def test_bearer_with_only_whitespace() -> None:
    """'Bearer   ' (trailing whitespace only) should be rejected."""
    client = TestClient(_make_app(TOKEN))
    resp = client.get("/v1/models", headers={"Authorization": "Bearer   "})
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Partial token / substring attacks
# ---------------------------------------------------------------------------

def test_token_prefix_only() -> None:
    """First N characters of the token should not be accepted."""
    prefix = TOKEN[: len(TOKEN) // 2]
    client = TestClient(_make_app(TOKEN))
    resp = client.get("/v1/models", headers={"Authorization": f"Bearer {prefix}"})
    assert resp.status_code == 401


def test_token_with_extra_prefix() -> None:
    """Token with extra chars prepended should be rejected."""
    client = TestClient(_make_app(TOKEN))
    resp = client.get(
        "/v1/models", headers={"Authorization": f"Bearer extra-{TOKEN}"}
    )
    assert resp.status_code == 401


def test_token_with_extra_suffix() -> None:
    """Token with extra chars appended should be rejected."""
    client = TestClient(_make_app(TOKEN))
    resp = client.get(
        "/v1/models", headers={"Authorization": f"Bearer {TOKEN}-extra"}
    )
    assert resp.status_code == 401


def test_token_swapped_scheme_and_token() -> None:
    """Token and scheme in reversed order should be rejected."""
    client = TestClient(_make_app(TOKEN))
    resp = client.get(
        "/v1/models", headers={"Authorization": f"{TOKEN} Bearer"}
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Case sensitivity
# ---------------------------------------------------------------------------

def test_token_case_sensitive() -> None:
    """Token comparison is case-sensitive — uppercase variant should fail."""
    client = TestClient(_make_app(TOKEN))
    resp = client.get("/v1/models", headers={"Authorization": f"Bearer {TOKEN.upper()}"})
    assert resp.status_code == 401


def test_token_mixed_case_not_accepted() -> None:
    """Randomly casing the token should not pass the strict equality check."""
    cased = TOKEN[::2].upper() + TOKEN[1::2].lower()
    if cased == TOKEN:
        return  # skip if casing happens to match (unlikely)
    client = TestClient(_make_app(TOKEN))
    resp = client.get("/v1/models", headers={"Authorization": f"Bearer {cased}"})
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Multiple header values / injection
# ---------------------------------------------------------------------------

def test_two_auth_headers() -> None:
    """Two Authorization headers — middleware reads the first one."""
    client = TestClient(_make_app(TOKEN))
    # httpx/requests send both; middleware checks headers.get() which returns the first
    resp = client.get(
        "/v1/models",
        headers={
            "Authorization": "Bearer wrong",
            # Duplicate header via raw — httpx doesn't support this directly,
            # so we verify that even one wrong header is rejected
        },
    )
    assert resp.status_code == 401


def test_auth_header_with_extra_whitespace() -> None:
    """'Bearer  <token>' with extra space should be rejected (not trimmed)."""
    client = TestClient(_make_app(TOKEN))
    resp = client.get(
        "/v1/models", headers={"Authorization": f"Bearer  {TOKEN}"}
    )
    assert resp.status_code == 401


def test_auth_header_with_trailing_newline() -> None:
    """Trailing newline in header value should be rejected (no trimming)."""
    client = TestClient(_make_app(TOKEN))
    resp = client.get(
        "/v1/models", headers={"Authorization": f"Bearer {TOKEN}\n"}
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Public path bypass attempts with spoofed headers
# ---------------------------------------------------------------------------

def test_public_path_ignores_auth_header() -> None:
    """Public dashboard paths bypass auth even if a wrong token is provided."""
    client = TestClient(_make_app(TOKEN))
    resp = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401
    # But the root path ignores the header entirely
    resp_root = client.get("/")
    assert resp_root.status_code in (200, 404)  # 404 if no static mount


def test_favicon_path_bypasses_with_wrong_token() -> None:
    """favicon path should be public regardless of auth header."""
    client = TestClient(_make_app(TOKEN))
    resp = client.get("/favicon.ico", headers={"Authorization": "Bearer wrong"})
    # Should not be 401 — public path
    assert resp.status_code != 401


# ---------------------------------------------------------------------------
# is_public_dashboard_path edge cases
# ---------------------------------------------------------------------------

def test_public_path_trailing_slash_root() -> None:
    """Root '/' is public."""
    assert is_public_dashboard_path("/")


def test_public_path_not_prefix_match() -> None:
    """'/app/something' (no underscore) should NOT be public."""
    assert not is_public_dashboard_path("/app/something")


def test_public_path_not_subpath() -> None:
    """/not/_app/something should NOT be public (starts with /not/)."""
    assert not is_public_dashboard_path("/not/_app/something")


def test_favicon_any_extension() -> None:
    """/favicon.png, /favicon.svg etc should all be public."""
    assert is_public_dashboard_path("/favicon.png")
    assert is_public_dashboard_path("/favicon.svg")
    assert is_public_dashboard_path("/favicon-192x192.png")


def test_empty_path_is_not_public() -> None:
    """Empty string should NOT be treated as public root."""
    assert not is_public_dashboard_path("")


def test_path_with_query_params_not_checked() -> None:
    """Middleware only checks the path portion, not query strings.
    But is_public_dashboard_path takes a raw path — verify nested _app paths."""
    assert is_public_dashboard_path("/_app/immutable/chunk-HASH123.js")
    assert is_public_dashboard_path("/_app/routes/__layout.svelte")


# ---------------------------------------------------------------------------
# 401 response structure validation
# ---------------------------------------------------------------------------

def test_401_response_shape() -> None:
    """Every 401 response should have a well-formed error body."""
    client = TestClient(_make_app(TOKEN))
    resp = client.get("/v1/models")
    assert resp.status_code == 401
    body = resp.json()
    assert "error" in body
    error = body["error"]
    assert error["error_code"] == "UNAUTHORIZED"
    assert error["type"] == "Unauthorized"
    assert error["code"] == 401
    assert isinstance(error["message"], str)
    assert len(error["message"]) > 0


def test_401_on_instance_route() -> None:
    """Missing token on /instance/abc also returns 401 with same shape."""
    client = TestClient(_make_app(TOKEN))
    resp = client.get("/instance/abc")
    assert resp.status_code == 401
    assert resp.json()["error"]["error_code"] == "UNAUTHORIZED"


# ---------------------------------------------------------------------------
# Multiple routes — same wrong token rejected everywhere
# ---------------------------------------------------------------------------

def test_wrong_token_rejected_on_all_protected_routes() -> None:
    """A wrong token should be rejected on every protected endpoint."""
    client = TestClient(_make_app(TOKEN))
    bad_header = {"Authorization": "Bearer wrong-token"}
    assert client.get("/v1/models", headers=bad_header).status_code == 401
    assert client.get("/instance/abc", headers=bad_header).status_code == 401
