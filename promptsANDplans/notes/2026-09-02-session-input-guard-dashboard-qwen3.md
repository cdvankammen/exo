# Session Notes: 2026-09-02 — Input Guard, Dashboard UX, Qwen3 Investigation

> **Session ID:** b42a5176-ba20-474d-a8ab-0d054e10e2c3
> **Branch:** fix-memory-error
> **Goal:** 461f5901-8913-42dc-970f-391a02312293 (comprehensive completion campaign)

## Commits This Session

| Hash | Description |
|------|-------------|
| `23843e9c` | fix(api): reject oversized inputs before they reach the runner (P0 #32, GitHub #560) |
| `093d7de7` | fix(dashboard): full-height sidebar logs + specific error cards |

## P0 #32: Input Length Guard

**Problem:** 66k-token inputs on a 24GB Mac Mini caused Metal OOM. The API layer has no tokenizer.

**Solution:** Character-count heuristic guard at the API layer:
- `EXO_MAX_INPUT_TOKENS` = 128000 (env-overridable)
- `EXO_CHARS_PER_TOKEN` = 4 (env-overridable)
- `_estimate_message_chars(messages: Sequence[object])` handles ChatCompletionMessage, ClaudeMessage, OllamaMessage, and raw dicts
- `_check_input_length()` raises `ApiError(status_code=400, error_code="INPUT_TOO_LONG")`
- Wired into all 4 text-gen endpoints: chat_completions, claude_messages, ollama_chat, ollama_generate

**Basedpyright journey:** Started with 16 errors → iterated through `list[Any]` → `Sequence[object]` + isinstance narrowing + targeted bare `# pyright: ignore` comments → 0 errors. Key lesson: `reportUnnecessaryTypeIgnoreComment = "error"` in this project means wrong rule names in ignore comments become errors themselves.

**Tests:** 13/13 pass in `test_input_length_guard.py` covering empty, short, multipart, pydantic model, boundary, and oversized inputs.

## Dashboard UX Fixes

### 1. Sidebar Logs Full Height
- `ChatSidebar.svelte`: Changed logs panel wrapper from `flex-shrink` to `flex-1 min-h-0`
- `ChatSidebar.svelte`: Conversation list capped at `max-h-[40vh]` so it doesn't consume all space
- `SidebarLogsPanel.svelte`: Replaced hardcoded `600px` pixel math with flex-based layout (`flex-1 overflow-y-auto` for errors, `flex-1 min-h-[40px]` for log tail)

### 2. Specific Error Cards
- `getInstanceRunnerError()` now returns `{errorMessage, diagnostics, runnerIds}`
- Both desktop (~line 5649) and mobile (~line 7021) instance failure cards updated:
  - Shows fallback message "Model failed to load — see logs tab for details" when `errorMessage` is null
  - Always displays runner ID for debugging
  - Error details visible even when backend did not populate `error_message`

### 3. Errors in Both Log Tabs
- Verified: SidebarLogsPanel and logs/+page.svelte both call `listLogErrors()` → same `/v1/logs/errors` API
- No data source gap exists; runner failures surface via instance cards with specific error info

## Qwen3-30B-A3B Load Failure Investigation

### Model Card Facts
- `mlx-community/Qwen3-30B-A3B-4bit`: **16.4 GB** storage, 48 layers, hidden_size=2048, num_kv_heads=4, supports_tensor=true
- `mlx-community/Qwen3.6-35B-A3B-4bit`: **19.0 GB** storage (different model!)
- User may have confused the two

### Placement Analysis
- Pipeline sharding on 4 nodes: ~4.1 GB/node (12 layers each) ✓ fits
- Tensor sharding on 4 nodes: ~4.1 GB/node (all layers, ¼ weights) ✓ fits
- Both modes valid per placement.py validation (hidden_size % 4 == 0, kv_heads % 4 == 0)

### Root Cause Identified
**Topology edge gaps.** The cluster state shows 6 nodes but the topology connections have `bandwidthMbps: null` for all links. Without stable edges:
- `topology.get_cycles()` returns only single-node cycles
- Pipeline/Tensor both require multi-node cycles
- Single-node fallback rewrites to Pipeline on 1 node → 16.4 GB exceeds 16 GB available → OOM

The election cycling every ~3 seconds (GitHub #2288) may be interfering with stable topology formation.

### Bandwidth Measurement Status
Bandwidth measurement IS fully built:
- `net_profile.py:58` `probe_bandwidth()`: HTTP GET to peer's `/v1/bandwidth-probe?size_bytes=1048576`
- `worker/main.py:492` `_poll_bandwidth_updates()`: runs every 30 seconds
- `topology.py:31` `bandwidth_mbps: float | None = None`
- Dashboard `TopologyGraph.svelte:424-444` renders "Xms YMB/s" labels

But `bandwidthMbps` is always null in current state → either poll hasn't fired or peer API unreachable.

## New Learnings

| # | Lesson |
|---|--------|
| L49 | basedpyright `reportUnnecessaryTypeIgnoreComment = "error"` means wrong rule names in `# pyright: ignore[RULE]` become errors. Use bare `# pyright: ignore` for duck-typed API data. |
| L50 | Topology edge gaps cause silent placement degradation: multi-node models fall back to single-node → OOM. Always check topology connections when debugging load failures. |
| L51 | Bandwidth measurement exists but has 30s poll interval + 15s timeout. Fresh clusters show null bandwidth for first ~45 seconds. |
| L52 | Election cycling (every ~3s) can prevent stable topology formation. Related to GitHub #2288. |

## Next Steps
1. Investigate why topology edges have null bandwidth / missing connections
2. Add CUDA OOM diagnostic classifier in `diagnostics.py`
3. Implement P0 #34 runner log rotation
4. Continue P1 quick wins
