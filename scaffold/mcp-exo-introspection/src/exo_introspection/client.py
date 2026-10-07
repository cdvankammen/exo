"""HTTP adapter for the exo API (the seam).

One class, small interface, all the transport complexity behind it: auth,
error mapping, retry-less read-only GETs, optional typed parse when the exo
package itself is importable.

The seam is real because two adapters exist today:
  1. raw JSON path  -- works without the exo package (used by the tests),
  2. typed path     -- ``State(**payload)`` when exo is importable,
both behind the same ``ExoClient`` interface.
"""

from __future__ import annotations

import os

import httpx

from .errors import (
    ExoAuthError,
    ExoConnectionError,
    ExoNotFoundError,
    ExoPlacementError,
    ExoProtocolError,
    ExoRateLimitError,
)

# No baked-in default port: a default that is wrong on the host you ship
# to is worse than none (the original 52415 default pointed at a
# dead/wedged listener). EXO_MCP_BASE_URL must be set per-host and
# verified against a live /state -> 200 before trusting it.
DEFAULT_BASE_URL = os.environ.get("EXO_MCP_BASE_URL")
DEFAULT_TIMEOUT_S = float(os.environ.get("EXO_MCP_TIMEOUT_S", "10.0"))

_MISSING_BASE_URL_MSG = (
    "EXO_MCP_BASE_URL is not set — refusing to guess an exo API port. "
    "Set it to the exo API origin for THIS host, e.g. "
    "EXO_MCP_BASE_URL=http://127.0.0.1:52415 (exo's default API port; "
    "override via EXO_API_PORT or --api-port). Verify before trusting: "
    "curl -s -m 3 -o /dev/null -w '%{http_code}' "
    "http://127.0.0.1:<port>/state must print 200 "
    "(a socket that is LISTENing in lsof but returns 000 is wedged — "
    "pick another port or restart exo)."
)


class ExoClient:
    """Read-only client for exo's HTTP API.

    Parameters
    ----------
    base_url:
        exo API origin, e.g. ``http://127.0.0.1:52415``. Required — falls
        back to ``EXO_MCP_BASE_URL``; raises ``ValueError`` if neither is
        set (no silent default port).
    api_token:
        Optional ``EXO_API_TOKEN``/tenant token; sent as ``Authorization:
        Bearer <token>``.
    timeout_s:
        Per-request timeout.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_token: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        resolved = base_url or os.environ.get("EXO_MCP_BASE_URL")
        if not resolved:
            raise ValueError(_MISSING_BASE_URL_MSG)
        self.base_url = resolved.rstrip("/")
        self.api_token = api_token or os.environ.get("EXO_MCP_API_TOKEN")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout_s,
            headers={
                "Accept": "application/json",
                # exo serves JSON for these routes regardless of Origin; an
                # Origin header is harmless and matches dashboard behavior.
                "Origin": self.base_url,
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get_state(self) -> dict:
        """GET /state -- full cluster snapshot (nodes, topology, downloads)."""
        return await self._get("/state")

    async def get_placement(self, **params) -> dict:
        """GET /instance/placement -- best placement for model_id."""
        return await self._get("/instance/placement", params=params)

    async def get_placement_previews(self, **params) -> dict:
        """GET /instance/previews -- candidate placements per strategy."""
        return await self._get("/instance/previews", params=params)

    async def get_node_compatibility(self, **params) -> dict:
        """GET /instance/node-compatibility -- per-node green/red verdicts."""
        return await self._get("/instance/node-compatibility", params=params)

    # ------------------------------------------------------------------ #

    async def _get(self, path: str, params: dict | None = None) -> dict:
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        try:
            resp = await self._client.get(path, params=params, headers=headers)
        except httpx.HTTPError as exc:
            raise ExoConnectionError(
                f"Could not reach exo API at {self.base_url}{path}: {exc}"
            ) from exc

        if resp.status_code == 401 or resp.status_code == 403:
            raise ExoAuthError(
                f"exo API rejected credentials ({resp.status_code}); "
                "set EXO_MCP_API_TOKEN or check tenant access"
            )
        if resp.status_code == 404:
            raise ExoNotFoundError(
                f"exo API returned 404 for {path} (unknown model/node/route)"
            )
        if resp.status_code == 429:
            raise ExoRateLimitError("exo API rate-limited (429); back off and retry")
        if resp.status_code == 400:
            # Placement failures arrive as 400 with error_code/detail.
            detail = _extract_error_detail(resp)
            raise ExoPlacementError(
                detail.get("detail") or f"exo API rejected request (400): {detail}",
                extra={"error_code": detail.get("error_code")},
            )
        if resp.status_code >= 500:
            raise ExoProtocolError(
                f"exo API server error {resp.status_code} for {path}: "
                f"{resp.text[:300]}"
            )
        try:
            payload = resp.json()
        except ValueError as exc:
            raise ExoProtocolError(
                f"exo API returned non-JSON for {path} ({resp.status_code}): "
                f"{resp.text[:200]!r}"
            ) from exc
        if not isinstance(payload, dict):
            raise ExoProtocolError(
                f"exo API returned unexpected shape for {path}: {type(payload).__name__}"
            )
        return payload


def _extract_error_detail(resp: httpx.Response) -> dict:
    try:
        data = resp.json()
    except ValueError:
        return {"detail": resp.text[:300]}
    if isinstance(data, dict):
        return data
    return {"detail": str(data)[:300]}
