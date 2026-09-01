# Upstream PRs, Guard Rails & Multi-Node Research
**Date:** 2026-09-01  
**Branch:** fix-memory-error (10+ commits ahead)  
**Upstream:** exo-explore/exo main

## Key Upstream Issues Found

### Pipeline Sharding & Placement
| # | Title | State | Relevance |
|---|-------|-------|-----------|
| #2196 | bf16 weights fail at inference under 3-node Pipeline sharding | OPEN | Quantized works, bf16 doesn't — same cluster |
| #957 | Better placement algorithm using memory bandwidth + latency | OPEN | Our `_cycle_accelerator_score` partially addresses this |
| #2077 | 2-node Mac cluster: asymmetric topology blocks placements | OPEN | Topology edge formation issues we've also hit |
| #1473 | `/place_instance` silently ignores sharding/instance_meta params | CLOSED | Fixed upstream — our fork already handles this |
| #1418 | Dynamic handling of prefill step size for pipeline parallel | CLOSED | Relevant to our ring attention work |
| #1740 | Multi-runner placement rollback not atomic after shard failure | CLOSED | Edge case we should watch |

### Ring Attention / Sequence Parallel
- **No open issues specifically about ring attention** in upstream exo
- Our `ring_attention.py` CUDA stream deadlock fix (`fabb605f`) is novel — upstream uses CPU streams which only work on Metal (unified memory)
- Commit `e9835615` (upstream) attempted a Fence::wait fix for Tensor/MlxJaccl hangs but introduced its own deadlock (#2208)

### GGUF / Non-MLX Backends
| # | Title | State |
|---|-------|-------|
| #1695 | [Feature Request] GGUF model loading | OPEN |
| #958 | Support all MLX backends | OPEN |
- No active GGUF integration work in upstream
- `Backend.Vllm` enum exists but no Vllm engine implementation
- GPUStack has llama.cpp custom backend integration — potential reference

### API Port / Multi-Instance Per Host
- **PR #1877** (merged): Thread `api_port` through Worker → reachability probes
  - This is exactly what our `net_profile.py` fix does independently
  - Upstream fixed it by passing `api_port` from Args; we additionally use `node_identities.api_port`
  - Our approach is more robust for multi-instance-per-host (tryingexo on pmox)

### Integration Tests
- **PR #1995** (merged): Full integration test infra with eco cluster lifecycle
  - 17 pytest tests across 5 files
  - Uses `EcoSession` for deploy/stop/start/release
  - We should adopt this pattern for our CI

## Guard Rail Audit (placement.py)

### All 15 ValueError Guards — force_override Status

| Line | Guard | force_override Before | After |
|------|-------|----------------------|-------|
| 217 | Ring requires MlxRing transport | ❌ Blocks | ✅ Bypassed |
| 219 | Ring requires min 2 nodes | ❌ Blocks | ✅ Bypassed |
| 221 | Model must declare Ring support | ❌ Blocks | ✅ Bypassed |
| 230 | Manual layers require Pipeline | ❌ Blocks | ⚠️ Kept (structural) |
| 238 | No cycle matches manual nodes | ❌ Blocks | ⚠️ Kept (structural) |
| 276 | No cycles with sufficient memory | ❌ Blocks | ✅ Falls back to all candidates |
| 280 | Tensor not supported by model | ✅ Already bypassed | ✅ |
| 297 | Tensor divisibility (hidden_size/kv_heads) | ✅ Already bypassed | ✅ |
| 308 | DeepSeek V3.1 Pipeline block | ✅ Already bypassed | ✅ |
| 320 | Gemma 4 single-node only | ✅ Already bypassed | ✅ |
| 332 | Backend incompatibility | ❌ Blocks | ✅ Bypassed |
| 354 | No RDMA cycles for Jaccl | ❌ Blocks | ✅ Bypassed |
| 383 | Single-node rewrite error | ❌ Blocks | ✅ Bypassed |
| 433 | (shard assignment errors) | N/A | N/A (runtime) |
| 556 | Instance not found (delete) | N/A | N/A (different path) |

### Design Philosophy
The user's position is correct: **exo's purpose is splitting models across nodes**. Guard rails should be advisory, not blocking. With `force_override=true`:
- Memory checks: bypassed (user accepts OOM risk)
- Backend checks: bypassed (user accepts possible crash)
- Model support checks: bypassed (user accepts undefined behavior)
- Structural checks (manual layers, node matching): kept (these are configuration errors, not safety)

## Node Compatibility Endpoint Math

### Pipeline Sharding (model split across N nodes)
```
per_node_bytes = storage_size / num_nodes
Example: Qwen3-30B-A3B-4bit (16.4GB) on 5 nodes = 3.28GB/node
```

### Tensor/Ring Sharding (replicated)
```
per_node_bytes = storage_size (full model on each node)
Example: Qwen3-30B-A3B-4bit (16.4GB) = 16.4GB/node
```

### Verified Behavior
- `curl ...?sharding=Pipeline`: tryingexo (14GB) ✅, mini3 (7GB) ✅, mini (3GB) ❌
- `curl ...?sharding=Tensor`: ALL nodes ❌ (16.4GB > any single node)
- `curl ...?force_override=true`: ALL nodes ✅ regardless of sharding

## Dashboard UX Improvements

### Sidebar Log Panel Height
- **Before:** Capped at 500px drag max, tail at `280 - errorsHeight` px
- **After:** No drag cap, tail at `max(40, 600 - errorsHeight)` px
- **Parent:** Changed from `flex-shrink-0` to `flex-shrink` so panel can grow

### Node Compatibility Pills
- Green/red pills per node in Load Model panel
- Hover tooltip shows full reason string
- Shows required model size and per-node split
- force_override flows through to endpoint

## Recommendations

1. **Adopt upstream integration test infra** (PR #1995) for CI
2. **Monitor #2196** — bf16 Pipeline failures may affect us
3. **GGUF support** is not coming from upstream soon — would need custom engine
4. **Our api_port fix is superior** to upstream's — we use node_identities for per-peer port resolution
5. **CUDA ring attention fix is novel** — consider upstreaming the `_is_cuda_backend()` stream selection
