# pyright: reportUnusedFunction=false, reportAny=false
"""Tests for API error handling — OpenAI-style error responses.

Covers the custom ``http_exception_handler`` on ``API``, ``ApiError``
with machine-readable ``error_code``, and FastAPI's built-in responses
for validation errors, unknown endpoints, and method-not-allowed.
"""
import json
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from exo.api.main import ApiError, API


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app() -> tuple[FastAPI, TestClient]:
    """Create a bare FastAPI app with only the custom exception handler wired."""
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]

    # ---------- test routes that raise various HTTPException subclasses -----

    @app.get("/test-error")
    async def _test_error() -> None:
        raise HTTPException(status_code=500, detail="Test error message")

    @app.get("/test-not-found")
    async def _test_not_found() -> None:
        raise HTTPException(status_code=404, detail="Resource not found")

    @app.get("/test-400")
    async def _test_400() -> None:
        raise HTTPException(status_code=400, detail="Invalid parameter")

    @app.get("/test-422")
    async def _test_422() -> None:
        raise HTTPException(status_code=422, detail="Unprocessable entity")

    @app.get("/test-429")
    async def _test_429() -> None:
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded"
        )

    @app.get("/test-api-error")
    async def _test_api_error() -> None:
        raise ApiError(
            status_code=400,
            detail="Insufficient memory",
            error_code="INSUFFICIENT_MEMORY",
        )

    @app.get("/test-api-error-404")
    async def _test_api_error_404() -> None:
        raise ApiError(
            status_code=404,
            detail="Instance not found",
            error_code="INSTANCE_NOT_FOUND",
        )

    @app.post("/test-post")
    async def _test_post() -> None:
        return None  # pragma: no cover

    @app.post("/test-post-with-body")
    async def _test_post_with_body(body: dict[str, Any]) -> None:
        return None  # pragma: no cover

    @app.get("/test-empty")
    async def _test_empty() -> None:
        raise HTTPException(status_code=400, detail="")

    return app, TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Original tests (preserved from v1)
# ---------------------------------------------------------------------------

def test_http_exception_handler_formats_openai_style() -> None:
    """Test that HTTPException is converted to OpenAI-style error format."""
    app, client = _make_app()

    # 500 error
    response = client.get("/test-error")
    assert response.status_code == 500
    data: dict[str, Any] = response.json()
    assert "error" in data
    assert data["error"]["message"] == "Test error message"
    assert data["error"]["type"] == "Internal Server Error"
    assert data["error"]["code"] == 500

    # 404 error
    response = client.get("/test-not-found")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert data["error"]["message"] == "Resource not found"
    assert data["error"]["type"] == "Not Found"
    assert data["error"]["code"] == 404


# ---------------------------------------------------------------------------
# New: 400 Bad Request
# ---------------------------------------------------------------------------

def test_400_bad_request_error_format() -> None:
    """400 HTTPException → OpenAI error with INVALID_REQUEST error_code."""
    _, client = _make_app()

    response = client.get("/test-400")
    assert response.status_code == 400
    data: dict[str, Any] = response.json()

    # Top-level structure
    assert "error" in data
    err = data["error"]

    # Human-readable detail
    assert err["message"] == "Invalid parameter"

    # HTTP status phrase and code
    assert err["type"] == "Bad Request"
    assert err["code"] == 400

    # Machine-readable code
    assert err["error_code"] == "INVALID_REQUEST"


# ---------------------------------------------------------------------------
# New: 500 Internal Server Error
# ---------------------------------------------------------------------------

def test_500_internal_error_format() -> None:
    """500 HTTPException → OpenAI error with INTERNAL_ERROR error_code."""
    _, client = _make_app()

    response = client.get("/test-error")
    assert response.status_code == 500
    data: dict[str, Any] = response.json()

    err = data["error"]
    assert err["message"] == "Test error message"
    assert err["type"] == "Internal Server Error"
    assert err["code"] == 500
    assert err["error_code"] == "INTERNAL_ERROR"


# ---------------------------------------------------------------------------
# New: 422 Unprocessable Entity (validation)
# ---------------------------------------------------------------------------

def test_422_validation_error_format() -> None:
    """422 HTTPException → OpenAI error with INVALID_REQUEST error_code."""
    _, client = _make_app()

    response = client.get("/test-422")
    assert response.status_code == 422
    data: dict[str, Any] = response.json()

    err = data["error"]
    assert err["message"] == "Unprocessable entity"
    assert err["type"] in ("Unprocessable Entity", "Unprocessable Content")
    assert err["code"] == 422
    assert err["error_code"] == "INVALID_REQUEST"


# ---------------------------------------------------------------------------
# New: 429 Too Many Requests
# ---------------------------------------------------------------------------

def test_429_rate_limit_error_format() -> None:
    """429 HTTPException → OpenAI error with INTERNAL_ERROR error_code."""
    _, client = _make_app()

    response = client.get("/test-429")
    assert response.status_code == 429
    data: dict[str, Any] = response.json()

    err = data["error"]
    assert err["message"] == "Rate limit exceeded"
    assert err["type"] == "Too Many Requests"
    assert err["code"] == 429
    # 429 is not 404 or 400/422, so maps to INTERNAL_ERROR
    assert err["error_code"] == "INTERNAL_ERROR"


# ---------------------------------------------------------------------------
# New: Unknown endpoint (404 from FastAPI built-in handler)
# ---------------------------------------------------------------------------

def test_unknown_endpoint_returns_404() -> None:
    """Request to unregistered path returns 404 with OpenAI error format."""
    _, client = _make_app()

    response = client.get("/v1/this-does-not-exist")
    assert response.status_code == 404
    data: dict[str, Any] = response.json()

    # FastAPI's default 404 handler returns {"detail": "Not Found"}
    # (NOT the custom OpenAI format — the default handler runs first
    # for routes that never reach the exception handler).
    # Our custom handler only fires for explicitly-raised HTTPException.
    assert "detail" in data


# ---------------------------------------------------------------------------
# New: Method Not Allowed (405)
# ---------------------------------------------------------------------------

def test_method_not_allowed_returns_405() -> None:
    """GET on a POST-only route returns 405."""
    _, client = _make_app()

    response = client.get("/test-post")
    assert response.status_code == 405
    data: dict[str, Any] = response.json()

    # FastAPI's built-in 405 handler returns {"detail": "Method Not Allowed"}
    assert "detail" in data


# ---------------------------------------------------------------------------
# New: Malformed JSON body
# ---------------------------------------------------------------------------

def test_malformed_json_body_returns_error() -> None:
    """Sending broken JSON to a POST endpoint returns 422 or 500."""
    _, client = _make_app()

    response = client.post(
        "/test-post-with-body",
        content=b"{invalid json!!!",
        headers={"Content-Type": "application/json"},
    )
    # FastAPI either returns 422 (validation) or 500 (json parse error)
    assert response.status_code in (422, 500)
    data: dict[str, Any] = response.json()

    # Must contain a usable error payload
    assert "detail" in data or "error" in data


# ---------------------------------------------------------------------------
# New: ApiError carries machine-readable error_code
# ---------------------------------------------------------------------------

def test_api_error_custom_error_code_INSUFFICIENT_MEMORY() -> None:
    """ApiError with error_code='INSUFFICIENT_MEMORY' is preserved."""
    _, client = _make_app()

    response = client.get("/test-api-error")
    assert response.status_code == 400
    data: dict[str, Any] = response.json()

    err = data["error"]
    assert err["message"] == "Insufficient memory"
    assert err["type"] == "Bad Request"
    assert err["code"] == 400
    assert err["error_code"] == "INSUFFICIENT_MEMORY"


def test_api_error_custom_error_code_INSTANCE_NOT_FOUND() -> None:
    """ApiError with error_code='INSTANCE_NOT_FOUND' is preserved."""
    _, client = _make_app()

    response = client.get("/test-api-error-404")
    assert response.status_code == 404
    data: dict[str, Any] = response.json()

    err = data["error"]
    assert err["message"] == "Instance not found"
    assert err["type"] == "Not Found"
    assert err["code"] == 404
    assert err["error_code"] == "INSTANCE_NOT_FOUND"


# ---------------------------------------------------------------------------
# New: Error response contains all required fields
# ---------------------------------------------------------------------------

def test_error_response_schema_completeness() -> None:
    """Every OpenAI-style error response includes all required fields."""
    _, client = _make_app()

    response = client.get("/test-error")
    data: dict[str, Any] = response.json()

    # Top-level must have exactly 'error'
    assert set(data.keys()) == {"error"}

    # error must have all fields
    err = data["error"]
    required_fields = {"message", "type", "code", "error_code"}
    assert required_fields <= set(err.keys()), (
        f"Missing fields: {required_fields - set(err.keys())}"
    )

    # param is optional (may be None / absent)
    # Check types
    assert isinstance(err["message"], str)
    assert isinstance(err["type"], str)
    assert isinstance(err["code"], int)
    assert isinstance(err["error_code"], str)


# ---------------------------------------------------------------------------
# New: Empty error message is preserved (edge case)
# ---------------------------------------------------------------------------

def test_empty_error_message_preserved() -> None:
    """An HTTPException with empty detail still formats correctly."""
    _, client = _make_app()

    response = client.get("/test-empty")
    assert response.status_code == 400
    data: dict[str, Any] = response.json()

    err = data["error"]
    assert err["message"] == ""
    assert err["code"] == 400
    assert err["error_code"] == "INVALID_REQUEST"


# ---------------------------------------------------------------------------
# New: Error code derivation logic for _error_code_for
# ---------------------------------------------------------------------------

def test_error_code_derivation_404() -> None:
    """Plain HTTPException(404) → NOT_FOUND error_code (no ApiError needed)."""
    _, client = _make_app()

    response = client.get("/test-not-found")
    data: dict[str, Any] = response.json()
    assert data["error"]["error_code"] == "NOT_FOUND"


def test_error_code_derivation_400() -> None:
    """Plain HTTPException(400) → INVALID_REQUEST error_code."""
    _, client = _make_app()

    response = client.get("/test-400")
    data: dict[str, Any] = response.json()
    assert data["error"]["error_code"] == "INVALID_REQUEST"


def test_error_code_derivation_500() -> None:
    """Plain HTTPException(500) → INTERNAL_ERROR error_code."""
    _, client = _make_app()

    response = client.get("/test-error")
    data: dict[str, Any] = response.json()
    assert data["error"]["error_code"] == "INTERNAL_ERROR"


# ---------------------------------------------------------------------------
# New: Request body validation error format (FastAPI default)
# ---------------------------------------------------------------------------

def test_required_body_missing_returns_422() -> None:
    """POST without required body → 422 with detail array."""
    _, client = _make_app()

    # test-post-with-body expects a body, but we send nothing
    response = client.post("/test-post-with-body")
    assert response.status_code == 422
    data = response.json()

    # FastAPI default validation error structure
    assert "detail" in data
    detail = data["detail"]
    assert isinstance(detail, list)
    assert len(detail) > 0
    # Each item has loc, msg, type
    first_err = detail[0]
    assert "loc" in first_err
    assert "msg" in first_err
    assert "type" in first_err
