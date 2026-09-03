# Session Notes: 2026-09-02 — Smox Node Isolation + Qwen3-30B-A3B Download Stall

> **Session ID:** b42a5176-ba20-474d-a8ab-0d054e10e2c3
> **Branch:** fix-memory-error
> **Goal:** 461f5901-8913-42dc-970f-391a02312293

## Issue 1: Smox Node Completely Isolated

### Symptoms
- Node "smox" (ID `4ab4c6aa...`) appears in topology with recent `lastSeen`
- Zero edges in `topology.connections` — no incoming or outgoing connections
- Other nodes cannot reach it via HTTP probes

### Root Cause
**Smox's exo API server is not running on port 52415.** Direct probe from this machine:
```
curl -s -m 3 http://10.2.0.76:52415/node_id → exit code 7 (connection refused)
```

The node joined zenoh gossip (hence appears in topology), but the HTTP API that other nodes use for reachability checks (`GET /node_id`) is down.

### Cluster Topology at Time of Investigation

| Node ID (short) | Hostname | OS | Backends | API Port | IPs |
|---|---|---|---|---|---|
| `12480a89` | tryingexo | Linux | MlxCpu, MlxCuda, Vllm | **52416** | 10.2.0.7 |
| `c48a0eed` | tryingexo | Linux | MlxCpu, MlxCuda, Vllm | **0** (unset) | 10.2.0.7 |
| `302cab18` | mini | macOS | MlxCpu, MlxMetal | 52415 | 10.2.0.52 |
| `872694d3` | mini3 | macOS | MlxCpu, MlxMetal | 52415 | 10.2.0.32 |
| `d3644e69` | mini2 | macOS | MlxCpu, MlxMetal | **0** (unset) | 10.2.0.51 |
| `4ab4c6aa` | **smox** | Linux | MlxCpu only | 52415 | 10.2.0.76, 192.168.1.76 |

### Additional Findings
- Two "tryingexo" nodes share identical IPs (same physical machine, two exo instances)
- `c48a0eed` and `d3644e69` have `apiPort: 0` — identity not fully advertised
- Smox has no Tailscale interface (other nodes do)

### Fix Steps
1. SSH into smox and check if exo is running: `ps aux | grep exo`
2. Check if API is bound: `ss -tlnp | grep 52415`
3. If not running, restart exo on smox
4. If running but bound to localhost, check `EXO_API_HOST` env var
5. Check firewall: `iptables -L -n | grep 52415` or `ufw status`

## Issue 2: Qwen3-30B-A3B Not Loading

### Placement Analysis
Placement **succeeds** — Pipeline sharding across 3 Metal nodes:
- Rank 0: `302cab18` (this Mac) — layers 0–9 (10 layers)
- Rank 1: `872694d3` — layers 9–37 (28 layers)
- Rank 2: `d3644e69` — layers 37–48 (11 layers)

Model card: `mlx-community/Qwen3-30B-A3B-4bit` = 17.6 GB, supports_tensor=true

### Why It Fails Despite Valid Placement

**Three compounding issues:**

1. **Download queue saturated**: Dozens of models queued on node `12480a89`, all with `downloaded: 0 bytes`. The download semaphore is full.

2. **GLM-4.7-8bit-gs32 causing repeated 404 crashes**: This model doesn't exist on HuggingFace. Every ~60 seconds it triggers `HuggingFaceNotFoundError`, consuming download slots and potentially blocking the queue.

3. **Election instability**: Master re-elected every 3 seconds (GitHub #2288). This may interrupt the placement→download→create flow before completion.

### Memory Pressure
- ct118 (`12480a89`): 89% pressure, only 1.7 GB available
- This Mac (`302cab18`): 82% pressure, only 2.8 GB available

### Fix Steps
1. Remove the non-existent model card:
   ```bash
   rm resources/inference_model_cards/mlx-community--GLM-4.7-8bit-gs32.toml
   ```
2. Restart exo to clear stalled downloads and election loop:
   ```bash
   pkill -f "uv run exo" && sleep 2 && uv run exo
   ```
3. Pre-download just the needed model:
   ```bash
   uv run python scripts/download_model_to_cluster.py mlx-community/Qwen3-30B-A3B-4bit
   ```
4. Launch with force override if memory is tight:
   ```bash
   curl -X POST "http://localhost:52415/instance/placement?model_id=mlx-community/Qwen3-30B-A3B-4bit&force_override=true&memory_tolerance=0.5"
   ```

## New Learnings

| # | Lesson |
|---|--------|
| L53 | A node can appear in topology (zenoh gossip) but have zero HTTP edges if its API server isn't running. Zenoh discovery ≠ HTTP reachability. Always verify `curl http://<node-ip>:<api-port>/node_id` when debugging isolated nodes. |
| L54 | A single non-existent model card (HF 404) can stall the entire download queue by repeatedly crashing the download pipeline. Remove bad cards immediately when seen in error logs. |
| L55 | Election cycling every 3s (GitHub #2288) can interrupt multi-step flows (placement→download→create). Models may fail to load even with valid placement if the master keeps re-electing. |
