# Placement Pipeline Exhaustive Audit — 2026-09-01

> Complete map of every filter, rejection, and override in the exo placement pipeline.
> Goal: understand why models fail to place, what can be overridden, and what needs fixing.

## Executive Summary

The placement pipeline has **14 distinct rejection stages** in `place_instance()` (`src/exo/master/placement.py`).
The dashboard already has node filtering (`previewNodeFilter`) and memory override (`memoryOverrideLevel` 0/1/2)
infrastructure wired through to the API. The backend API (`/instance/previews`) already accepts `node_ids`,
`force_override`, and `memory_tolerance` query parameters.

**Key fixes applied this session:**
- `force_override` now bypasses Gemma 4 single-node restriction
- `force_override` now bypasses Tensor divisibility checks (hidden_size, kv_heads)
- `force_override` now bypasses DeepSeek V3.1 8-bit Pipeline block
- `force_override` now bypasses strict Ring memory admission (4× KV working set estimate)

## The 14 Rejection Stages

### Stage 1: Ring Transport Check (line ~215)
```
if sharding is Ring and instance_meta is not MlxRing → ValueError
```
**Override:** None (hard requirement — Ring attention requires MlxRing transport)

### Stage 2: Ring Min Nodes (line ~218)
```
if sharding is Ring and min_nodes < 2 → ValueError
```
**Override:** None (Ring requires ≥2 nodes by definition)

### Stage 3: Ring Model Support (line ~220)
```
if sharding is Ring and not model_card.supports_ring → ValueError
```
**Override:** None (model must declare Ring support in its card)

### Stage 4: Manual Layer Allocation (line ~228)
```
if node_layers is not None and sharding != Pipeline → ValueError
if set(cycle.node_ids) != requested_nodes → ValueError
```
**Override:** None (exact match required for manual allocation)

### Stage 5: Required Nodes Subset (line ~244)
```
if required_nodes and not required_nodes.issubset(cycle.node_ids) → filtered out
```
**Override:** This IS the node selection mechanism — user picks nodes via `node_ids` API param

### Stage 6: Memory Filter (line ~248)
- **Ring:** `filter_cycles_by_replicated_memory()` — EVERY node must hold full model + KV working set
- **Pipeline/Tensor:** `filter_cycles_by_memory()` — SUM of available RAM across cycle ≥ storage_size

**Override:** `force_override=True` bypasses entirely; `memory_tolerance<1.0` relaxes proportionally
**FIXED:** Ring now also respects `force_override` (previously ignored it)

### Stage 7: Tensor Divisibility (line ~268)
```
hidden_size % len(cycle) == 0 AND (kv_heads is None OR kv_heads % len(cycle) == 0)
```
**Override:** `force_override=True` now skips this check entirely
**Note:** DeepSeek V4 exempted (MQA with MoE expert sharding)

### Stage 8: DeepSeek V3.1 Pipeline Block (line ~293)
```
if Pipeline and model_id == "mlx-community/DeepSeek-V3.1-8bit" → ValueError
```
**Override:** `force_override=True` now bypasses this block

### Stage 9: Gemma 4 Single-Node Restriction (line ~300)
```
if Pipeline and base_model.startswith("Gemma 4") → filter to len(cycle)==1 only
```
**Override:** `force_override=True` now bypasses — allows multi-node Pipeline for Gemma 4
**Risk:** May not work on all Gemma 4 variants; user assumes risk

### Stage 10: Smallest Cycle Selection (line ~310)
```
get_smallest_cycles() → returns only cycles with min(len(cycle))
```
**Override:** NONE — this is why users can't choose larger cycles
**Impact:** If 2-node and 4-node cycles both pass memory, only 2-node is considered
**TODO:** Add `max_nodes` or `preferred_node_count` parameter

### Stage 11: Backend Matching (line ~312)
```
required_backends = INSTANCE_META_BACKENDS[instance_meta] ∩ model_card.backends
Every node in cycle must support at least one required backend
```
**Override:** None (hard capability constraint — can't run Metal on CPU-only node)
**Example:** smox (MlxCpu only) excluded from MlxJaccl (requires MlxMetal)

### Stage 12: RDMA Cycle Check for MlxJaccl (line ~350)
```
if instance_meta == MlxJaccl and no RDMA-connected cycles → ValueError
```
**Override:** None (RDMA is hardware requirement for Jaccl)

### Stage 13: Leaf Node Preference (line ~370)
```
cycles_with_leaf_nodes preferred over non-leaf cycles
```
**Override:** Automatic — falls back to non-leaf if no leaf cycles exist

### Stage 14: Cycle Scoring & Selection (line ~380)
```
max(accelerator_score, download_score, total_available_memory)
```
**Override:** Automatic — picks best cycle by scoring heuristic

## Dashboard Infrastructure Already Present

### Node Filtering
- Store: `previewNodeFilter: Set<string>` (app.svelte.ts line ~693)
- Toggle: `togglePreviewNodeFilter(nodeId)` (line ~1740)
- Clear: `clearPreviewNodeFilter()` (line ~1752)
- Topology graph: `filteredNodes` prop + `onNodeClick` handler (lines 5075-5076)
- Filter indicator: "FILTER: N" badge with clear button (lines 5309-5330)
- API wiring: `fetchPlacementPreviews` sends `&node_ids=` for each filtered node (line ~1682)

### Memory Override (3 levels)
- Level 0: Off (strict memory checks)
- Level 1: Relaxed (`memory_tolerance` parameter, e.g. 0.5 = admit cycles holding ≥50% of model)
- Level 2: Force (`force_override=true`, bypasses ALL memory checks)
- Settings UI: `/settings` page has 3-button selector + tolerance slider
- API wiring: `fetchPlacementPreviews` sends appropriate params (lines 1674-1680)

### What's Missing
1. **Per-node feasibility indicators** — no green/red status per node in model picker
2. **Force override accessibility** — buried in Settings, should be near model picker
3. **Cycle size selection** — `get_smallest_cycles()` auto-picks smallest, no user control
4. **Node compatibility endpoint** — no API to ask "why can't this node run this model?"

## Live Cluster State (2026-09-01)

| Node | ID Prefix | Backends | Available RAM | Interfaces | Out Edges |
|------|-----------|----------|---------------|------------|-----------|
| smox | 9fda5ae9 | [MlxCpu] | 11.6 GB | 6 | 6 |
| mini | 1b811a97 | [MlxCpu, MlxMetal] | 2.9 GB | 26 | 10 |
| mini3 | 872694d3 | [MlxCpu, MlxMetal] | 10.1 GB | 30 | 10 |
| (4th) | d3644e69 | [MlxCpu, MlxMetal] | 2.6 GB | 27 | 12 |

**Total directed edges:** 38 (fully connected mesh)
**smox issue:** Only advertises MlxCpu — excluded from MlxMetal placements

## Why "No Valid Configurations" Happens

For Qwen3-30B-A3B-4bit (17.6 GB storage):
- ✅ MlxRing Pipeline works (2-3 node cycles have enough combined RAM)
- ❌ MlxJaccl fails: "No cycle where every node supports MlxMetal" (smox = MlxCpu only)
- ⚠️ Some 2-node cycles fail memory: mini (2.9GB) + d3644e69 (2.6GB) = 5.5GB < 17.6GB

For Gemma 4 31B:
- ❌ Pipeline forced to single-node (Stage 9) — no single node has enough RAM
- ✅ With `force_override=True`: multi-node Pipeline now allowed (fixed this session)

## Recommendations

1. **Add `/instance/node-compatibility` endpoint** — returns per-node status (ok/warning/error) with reason strings
2. **Add `preferred_node_count` to PlaceInstance** — let user request specific cycle size instead of auto-smallest
3. **Move force override toggle to model picker** — make it visible alongside sharding/runtime selectors
4. **Fix smox backend detection** — investigate why Linux/AMD container only reports MlxCpu
5. **Add per-node memory pressure display** — show real-time available RAM in node picker
