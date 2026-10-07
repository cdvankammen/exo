"""MCP server exposing read-only exo cluster introspection tools.

Four tools, one small interface (see specs/mcp-exo-introspection.md):

  * list_nodes            -- nodes in the cluster, with optional dynamic state
  * get_topology          -- cluster graph: nodes, links (socket/RDMA), cycles
  * explain_placement     -- why model X would land on nodes Y (placement scoring)
  * get_download_status   -- per-node model download progress

Everything is read-only: we only GET exo's JSON API and normalize the
responses. ``EXO_MCP_BASE_URL`` is required — there is deliberately no
baked-in default port (exo's own default is 52415, but it must be
verified per-host, see README).

Run:
    EXO_MCP_BASE_URL=http://127.0.0.1:<verified-port> python -m exo_introspection
"""

from .server import create_server, main

__all__ = ["create_server", "main"]
__version__ = "0.1.0"