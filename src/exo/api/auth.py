"""Opt-in API bearer-token authentication (T23).

When ``EXO_API_TOKEN`` is set at import time, the API installs
:func:`api_token_auth_middleware` requiring ``Authorization: Bearer
<EXO_API_TOKEN>`` on every route except the dashboard static assets
(``/``, ``/_app/*``, favicon) so the UI still loads.

Kept in its own module so it is unit-testable without importing
``exo.api.main`` (which pulls in the full API surface).
"""

from typing import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse


def is_public_dashboard_path(path: str) -> bool:
    """Dashboard static assets stay public; everything else requires the token."""
    return path == "/" or path.startswith(("/_app/", "/favicon"))


def api_token_auth_middleware(token: str):
    """Return a FastAPI http middleware requiring ``Authorization: Bearer <token>``."""

    async def middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[StreamingResponse]],
    ) -> StreamingResponse | JSONResponse:
        if is_public_dashboard_path(request.url.path):
            return await call_next(request)
        if request.headers.get("authorization") != f"Bearer {token}":
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Unauthorized. Set the Authorization: Bearer <EXO_API_TOKEN> header.",
                        "type": "Unauthorized",
                        "code": 401,
                        "error_code": "UNAUTHORIZED",
                    }
                },
            )
        return await call_next(request)

    return middleware
