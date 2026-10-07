# exo-introspection MCP server

Read-only MCP server exposing exo cluster introspection as four tools:

| Tool | Answers |
| --- | --- |
| `list_nodes` | What nodes are in the cluster (optional live memory/backends/disk detail) |
| `get_topology` | Cluster graph: nodes, socket/RDMA links, latency/bandwidth, optional RDMA cycles |
| `explain_placement` | How exo would place a model: winning cycle, per-node shard assignments, score factors (or per-strategy failure reasons) |
| `get_download_status` | Per-node model download progress (pending/ongoing/completed/failed/stalled, percent, speed, ETA) |

Everything is **read-only** — the server only GETs the exo HTTP API
(`/state`, `/instance/placement`, `/instance/previews`). No POSTs, no
mutation.

Grounded in exo `src/exo/` (branch `fix-memory-error`): state models
`src/exo/shared/types/state.py`, tagged-union `DownloadProgress`
(`src/exo/shared/types/worker/downloads.py`), placement pipeline
(`src/exo/master/placement.py` + `placement_utils.py`), API routes
(`src/exo/api/main.py`, `src/exo/api/types/api.py`).

Design spec: `hermesResearch/Exo-research/V2/specs/mcp-exo-introspection.md`

## Requirements

- Python 3.11+
- `mcp>=1.0,<2` (FastMCP API; mcp 2.x renamed FastMCP → MCPServer)
- `httpx`, `pydantic>=2.9`

## Install

```bash
cd scaffold/mcp-exo-introspection
python -m venv .venv && . .venv/bin/activate
pip install -e .
```

Or reuse the exo repo venv (verified working):

```bash
cd /Users/chris/Documents/GitHub/exo
uv pip install --python .venv/bin/python 'mcp>=1.0,<2' httpx
```

## Run

`EXO_MCP_BASE_URL` is **required** — there is deliberately no baked-in
default port (a default that is wrong on the host you ship to is worse
than none). exo's own default API port is **52415** (`EXO_API_PORT` /
`--api-port` override it), but the port must be **verified per-host**:
multiple instances have been observed on 52515/52815/52925, and a bound
but wedged listener accepts nothing (curl returns `000` while `lsof`
shows `LISTEN`).

Verify before pointing the MCP server at a port:

```bash
# what is actually bound right now:
lsof -nP -iTCP -sTCP:LISTEN | grep -E ':5[0-9]{4} '
# must print 200 — 000 while lsof shows LISTEN means wedged listener:
curl -s -m 3 -o /dev/null -w '%{http_code}\n' http://127.0.0.1:<port>/state
```

The server refuses to start without `EXO_MCP_BASE_URL` and prints these
instructions instead of silently falling back to a stale port.

```bash
export EXO_MCP_BASE_URL=http://127.0.0.1:<verified-port>
# optional when exo runs with API token auth:
export EXO_MCP_API_TOKEN=...
python -m exo_introspection                        # stdio MCP transport
```

_Port guidance last verified 2026-10-07 on mini.local: no exo instance
was running fleet-wide at verification time, so the fail-fast path (tests)
is the verified behavior; re-run the curl ladder once a cluster is up._

## Test

```bash
python -m pytest src/exo_introspection/tests -q
# 43 passed (uses httpx MockTransport — no cluster needed)
```

## Register with Hermes

Point Hermes at the server via `mcp.servers` in `config.yaml`:

```yaml
mcp:
  servers:
    exo-introspection:
      command: /Users/chris/Documents/GitHub/exo/.venv/bin/python
      args: ["-m", "exo_introspection"]
      env:
        EXO_MCP_BASE_URL: "http://127.0.0.1:<verified-port>"   # see Run — verify first
```

(Equivalent JSON registration: `config/exo-introspection.json`. That file
ships with an empty `EXO_MCP_BASE_URL` on purpose — fill in your verified
port before registering; the server exits with instructions if it is
blank.)

## Behavior notes (read tool descriptions for full contract)

- **Node IDs are not stable**: 16-byte hex, regenerated per process start.
  Correlate by `friendly_name` / `api_host:api_port`.
- **Downloads are per node**: `state.downloads` is keyed by node; a model
  downloaded on node A does not imply node B has it.
- **Placement is master-authoritative**: a non-master node may report stale
  topology; point `EXO_MCP_BASE_URL` at the master.
- **Placement failure is a rich result**: `explain_placement` falls back to
  `/instance/previews` and returns `{"ok": true, "result": {"ok": false,
  "error": {code: PLACEMENT_FAILED, detail}, "candidates": [...]}}` — never a
  bare error.

## Layout

```
src/exo_introspection/
├── server.py      # FastMCP wiring — the 4-tool interface (thin)
├── client.py      # ExoClient — httpx adapter at the seam (auth, error mapping)
├── normalize.py   # /state JSON → flat tool shapes (tagged-union decode)
├── explain.py     # placement JSON → readable reason/score/shard summary
├── errors.py      # ExoError hierarchy → {"ok": false, "error": {...}}
└── tests/         # 27 tests, httpx MockTransport fixtures
```