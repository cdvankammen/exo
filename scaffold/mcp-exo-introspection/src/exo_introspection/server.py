"""FastMCP server: four read-only exo introspection tools.

The interface is deliberately small: every tool takes a flat JSON object of
primitives and returns ``{"ok": true, "result": ...}`` or
``{"ok": false, "error": {...}}``. All logic lives in client.py /
normalize.py / explain.py; this file only wires them together.
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from .client import ExoClient, DEFAULT_BASE_URL
from .errors import ExoError, ExoPlacementError
from .explain import build_reason, summarize_placement, summarize_placement_failure
from .normalize import (
    normalize_downloads,
    normalize_nodes,
    normalize_topology,
    state_map,
)

_SERVER_NAME = "exo-introspection"


def _ok(result: dict | list) -> dict:
    return {"ok": True, "result": result}


def _err(error: ExoError) -> dict:
    return {"ok": False, "error": error.to_payload()}


def create_server(base_url: str | None = DEFAULT_BASE_URL) -> FastMCP:
    """Build the MCP server bound to one ExoClient.

    Pass base_url explicitly in tests; otherwise ``EXO_MCP_BASE_URL`` is
    required — there is deliberately no baked-in default port (verify the
    port per-host first, see README).
    """
    app = FastMCP(_SERVER_NAME)
    client = ExoClient(base_url)

    @app.tool()
    async def list_nodes(include_state: bool = False) -> dict:
        """List the nodes currently in the exo cluster.

        Returns one entry per connected node. Node IDs are random 16-byte
        hex per process start and change on restart — never persist them;
        correlate by friendly_name / api_host:api_port instead.

        include_state=true adds dynamic per-node detail (memory, backends,
        disk, rdma, os) from the live cluster snapshot; default false keeps
        the result cheap and stable.

        Read-only. Never mutates cluster state.
        """
        try:
            state = await client.get_state()
            return _ok({"nodes": normalize_nodes(state, include_state), "count": len(
                (state.get("topology") or {}).get("nodes", [])
            )})
        except ExoError as exc:
            return _err(exc)

    @app.tool()
    async def get_topology(
        include_cycles: bool = False,
        include_node_details: bool = False,
    ) -> dict:
        """Return the exo cluster graph: nodes, links, optional cycles.

        Edges list every socket/RDMA connection between peers with latency
        and bandwidth when reported. include_cycles=true adds the RDMA
        cycle list (only meaningful for ring/RDMA placement reasoning);
        include_node_details=true embeds the per-node attendee summary
        (friendly name + api host/port, when known).

        Read-only.
        """
        try:
            state = await client.get_state()
            result = normalize_topology(state, include_cycles)
            if include_node_details:
                node_ids = result.get("nodes", [])
                # Same wire-name rule as normalize_nodes: State serializes by
                # camelCase alias, and apiHost/apiPort live on nodeIdentities —
                # a live payload has no nodeApiInfo map at all.
                identities = state_map(state, "nodeIdentities", "node_identities")
                api_info = state_map(state, "nodeApiInfo", "node_api_info")
                result["node_details"] = [
                    {
                        "node_id": nid,
                        "friendly_name": (identities.get(nid) or {}).get(
                            "friendlyName", (identities.get(nid) or {}).get("friendly_name")
                        ),
                        "api_host": (
                            (identities.get(nid) or {}).get(
                                "apiHost", (identities.get(nid) or {}).get("api_host")
                            )
                            or (api_info.get(nid) or {}).get(
                                "apiHost", (api_info.get(nid) or {}).get("api_host")
                            )
                        ),
                        "api_port": (
                            (identities.get(nid) or {}).get(
                                "apiPort", (identities.get(nid) or {}).get("api_port")
                            )
                            or (api_info.get(nid) or {}).get(
                                "apiPort", (api_info.get(nid) or {}).get("api_port")
                            )
                        ),
                    }
                    for nid in node_ids
                ]
            return _ok(result)
        except ExoError as exc:
            return _err(exc)

    @app.tool()
    async def explain_placement(
        model_id: str,
        sharding: str = "Pipeline",
        instance_meta: str = "MlxRing",
        min_nodes: int = 1,
        memory_tolerance: float = 1.0,
        force_override: bool = False,
    ) -> dict:
        """Explain how exo would place a model: which nodes, which shards, why.

        Calls exo's own placement scoring (GET /instance/placement) and
        summarizes the winning cycle: shard assignments per node (layer
        ranges, device ranks) and the score factors that picked it.

        sharding: Pipeline (splits the model across nodes) | Tensor | Ring
        (replicates it). instance_meta: MlxRing | MlxJaccl.
        memory_tolerance 0..1 relaxes the memory-sufficiency check; use
        force_override=true only to bypass guardrails — the response then
        lists exactly which guardrails were bypassed.

        On failure returns {"ok": false, "error": {code: PLACEMENT_FAILED,
        detail: <exo's own explanation>}} plus per-strategy candidates from
        /instance/previews when available.

        Read-only — nothing is started or downloaded.
        """
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
            return _ok(
                summarize_placement_failure(model_id, exc.to_payload(), previews)
            )
        except ExoError as exc:
            return _err(exc)

        summary = summarize_placement(placement)
        if not summary.get("reason"):
            summary["reason"] = build_reason(placement, sharding, min_nodes)
        return _ok(summary)

    @app.tool()
    async def get_download_status(
        model_id: str | None = None,
        status: str | None = None,
    ) -> dict:
        """Report model download progress per node in the exo cluster.

        One entry per node x model. status filter: pending | ongoing |
        completed | failed | stalled (stalled = no progress for a while,
        retryable). Omit model_id for all downloads. Percent is computed
        from bytes; speed/eta appear for ongoing downloads.

        Downloads are per-node: a model downloaded on node A does not imply
        it exists on node B. Read-only.
        """
        try:
            state = await client.get_state()
            rows = normalize_downloads(
                state, model_id=model_id, status=status
            )
            return _ok(
                {
                    "downloads": rows,
                    "count": len(rows),
                    "summary": _download_summary(rows),
                }
            )
        except ExoError as exc:
            return _err(exc)

    return app


def _download_summary(rows: list[dict]) -> dict:
    """One-glance totals across the download rows."""
    summary: dict = {"by_status": {}}
    for row in rows:
        st = row["status"]
        summary["by_status"][st] = summary["by_status"].get(st, 0) + 1
    total_bytes = sum(r["total_bytes"] for r in rows if r.get("total_bytes"))
    downloaded_bytes = sum(
        r["downloaded_bytes"] for r in rows if r.get("downloaded_bytes")
    )
    if total_bytes:
        summary["overall_percent"] = round(100.0 * downloaded_bytes / total_bytes, 2)
    return summary


def main() -> int:
    """Run the server over stdio (MCP transport)."""
    app = create_server()
    app.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())