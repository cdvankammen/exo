# pyright: reportUnusedFunction=false, reportAny=false
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from exo.api.main import API


def test_http_exception_handler_formats_openai_style() -> None:
    """Test that HTTPException is converted to OpenAI-style error format."""

    app = FastAPI()

    # Setup exception handler
    api = object.__new__(API)
    api.app = app
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]

    # Add test routes that raise HTTPException
    @app.get("/test-error")
    async def _test_error() -> None:
        raise HTTPException(status_code=500, detail="Test error message")

    @app.get("/test-not-found")
    async def _test_not_found() -> None:
        raise HTTPException(status_code=404, detail="Resource not found")

    client = TestClient(app)

    # Test 500 error
    response = client.get("/test-error")
    assert response.status_code == 500
    data: dict[str, Any] = response.json()
    assert "error" in data
    assert data["error"]["message"] == "Test error message"
    assert data["error"]["type"] == "Internal Server Error"
    assert data["error"]["code"] == 500

    # Test 404 error
    response = client.get("/test-not-found")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert data["error"]["message"] == "Resource not found"
    assert data["error"]["type"] == "Not Found"
    assert data["error"]["code"] == 404
    assert data["error"]["error_code"] == "NOT_FOUND"


def test_api_error_carries_stable_code() -> None:
    """ApiError with explicit error_code surfaces it in the JSON response."""
    from exo.api.main import ApiError

    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]

    @app.get("/test-oom")
    async def _test_oom() -> None:
        raise ApiError(
            status_code=400,
            detail="Not enough VRAM",
            error_code="INSUFFICIENT_MEMORY",
        )

    @app.get("/test-input-too-long")
    async def _test_input() -> None:
        raise ApiError(
            status_code=400,
            detail="Input too long",
            error_code="INPUT_TOO_LONG",
        )

    client = TestClient(app)

    resp = client.get("/test-oom")
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["error_code"] == "INSUFFICIENT_MEMORY"
    assert body["error"]["code"] == 400

    resp = client.get("/test-input-too-long")
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["error_code"] == "INPUT_TOO_LONG"
