1. ~~EXO_BOOTSTRAP_PEERS is currently broken~~ → **FIXED (T21)** — parsed from env in `src/exo/main.py` (`--bootstrap-peers`/`EXO_BOOTSTRAP_PEERS`); `EXO_ZENOH_CONNECT` unicast dial (`d300410f`, PR #2243) covers cross-subnet without multicast

4. ~~I'd like to see profiled network latency / bandwidth.~~ → **DONE** — topology links show measured latency (reachability probes) + bandwidth (HTTP probe): `graph-link-latency` labels (`234c322a`).
5. ~~I'd like to see how much bandwidth each link is using.~~ → **DONE** — per-link `bandwidth_mbps` shown on topology edges (`234c322a`).
7. ~~In continuous batching, a new prompt blocks decode of the current batch until prefill completes~~ → **DONE (`24ec252c`)** — `batch_generate.py` defers prefill to the chunked decode-first path when possible (`can_defer_prefill()` guard + 6 tests in `test_deferred_prefill.py`)
8. ~~Offline model copy detection~~ → **DONE** — non-empty local model dir with no .partial files detected as fully-downloaded (`download_utils.py:429`).
13. ~~Memory pressure instead of memory used~~ → **DONE** — `profiling.py` computes `pressure` (used/total) on RAM + VRAM, surfaced in node cards.
14. ~~Show the type of each connection (TB5, Ethernet, etc.) in the UI~~ → **DONE** — `interface_type` + `link_speed_megabits` in `NodeNetworkInfo`; TB4/TB5 distinction via link speed.
16. Dynamically switch to higher priority connection when it becomes available. Probably bring back InstanceReplacedAtomically.
17. Faster model loads by streaming model from other devices in cluster.
18. ~~Add support for specifying the type of network connection to use in a test~~ → **DONE** — `@pytest.mark.cluster(..., thunderbolt='a2a')` in `tests/framework.py`, used by test_2node/resilience.
27. ~~Log cleanup - per-module log filters and default to DEBUG log levels~~ → **DONE** — per-module filters via `EXO_LOG_LEVELS` (`91d60a01`) + rotation for `.exo/exo.log` (`009b43c6`)
28. ~~Validate RDMA connections with ibv_devinfo in the info gatherer~~ → **DONE (`383ac419`)** — `_gather_has_verbs_device()` runs `ibv_devices` in `utils/info_gatherer/info_gatherer.py`; placement/apply gate on `has_verbs_device` (`test_rdma_verbs_validation.py`)

---

## Master Plan Sprint Items (from `research/master-implementation-plan-2026-09-02.md`)

### P0 — Stability (do first)
~~29. Guard `publish_bytes` against malformed events (malformed event kills process → exit 1)~~ → **DONE (`ef16c380`)** — `publish_bytes` wraps `topic.deserialize` in try/except at `router.py:109-120`; malformed events log warning + `record_malformed_event` for dashboard chip instead of crashing process
~~30. Pre-prefill eviction in `cache.py` (OOM during prefill, #2182)~~ → **DONE (`5260b384`)** — `evict_for_prefill()` at `cache.py:691` evicts LRU entries below `_PREFILL_MEMORY_THRESHOLD` before forward pass
~~31. Periodic cache cleanup + gc.collect() (idle memory leak, #2262)~~ → **DONE** — `gc.collect()` + `mx.clear_cache()` called after every eviction at `cache.py:721-722`; `_janitor_sweep_stale_slots()` at line 805 sweeps orphaned KV slot dirs on disk
~~32. Input length limiter (66k-token OOM, #560)~~ → **DONE** — `EXO_MAX_INPUT_TOKENS` constant (default 128000) in `shared/constants.py:127`; enforced at `api/main.py:421-428` with 400 response on oversized input
~~33. Node identity periodic re-announce (lost startup announcement → invisible node)~~ → **DONE (`dead4e88`)** — `_monitor_node_config` re-sends identity on interval; companion `9299909c` does same for NodeBackends; both tested in `test_node_config.py`
34. ~~Log rotation for runner stdout/stderr (Linux log grew to 14GB → disk full)~~ → **DONE (`57eab048`)** — runner subprocess logs rotate; complements main `exo.log` rotation (`009b43c6`)

### P1 — Quick Wins (week 1-2)
~~35. Bandwidth-aware pipeline placement (#957 water-filling algorithm)~~ → **DONE (`78f7fa8e`+`35153898`)** — `WaterFillingAllocator` in `master/placement.py` allocates layers by per-link bandwidth; duplicates diagnosed #66
~~36. Ring context configurable (`EXO_RING_ADMISSION_CONTEXT` env var)~~ → **DONE (`d0cee8c1`)** — read at call time in `placement_utils.py`; testable via monkeypatch
~~37. TP single-node heuristic (don't split if one node fits)~~ → **DONE (`df7f0aa9`)** — Pipeline cycle prefers single-node when one node has sufficient memory; duplicates diagnosed #67
~~38. Pipeline activation memory estimate (~10% of weights)~~ → **DONE (`c4165eee`)** — `EXO_ACTIVATION_MEMORY_FRACTION` reserve in `placement.py`; duplicates diagnosed #68
~~39. OOM graceful degradation (evict → halve batch → retry instead of SIGABRT)~~ → **DONE (`270b61c4`)** — runner evicts cache + retries once on OOM
~~40. Auto-restart VRAM check (skip restart if memory >90%)~~ → **DONE (`619d6946`)** — skip runner restart when memory >90%
~~41. BUG3: tryingexo nodes outgoing edges (deploy latest containers, verify cycles)~~ → **VERIFIED (86f9f246)** — tryingexo nodes have outgoing edges forming 2-cycles in mesh topology. Live cluster data confirmed bidirectional edges via multi-interface (tailscale, LAN, loopback). No code defect.

### P2 — Core Performance (weeks 2-4)
42. Continuous batching scheduler (2-4× throughput under load)
43. PagedAttention-style KV cache (2× memory efficiency)
44. Speculative decoding (draft on small node, target on big, 2-3× decode)

### P3 — Cluster Intelligence (weeks 4-6)
45. Cluster-wide radix prefix cache (3× TTFT for repeated prompts)
46. KV cache offload GPU→CPU→disk (2× concurrency)
47. Chunked prefill — interleave prefill/decode (no decode blocking)
48. Topology-aware placement (use bandwidth probe data)

### P4 — Production Readiness (weeks 6-8)
49. Per-request metrics + `/metrics` Prometheus endpoint
50. Multi-tenancy (API keys + quotas)
51. Load balancing across instances
52. Graceful degradation (circuit breaker, instance migration)

### Dashboard/UX Backlog
~~53. Logs page: per-node + aggregated "all nodes" view~~ → **DONE (`f1611a55`)** — node filter chips on errors view parse source_log prefix (`node_id::`) to identify remote nodes, resolve to friendly names via nodeIdentities, All/per-node filter with counts
54. ~~Error table on Logs page (structured WARNING/ERROR list)~~ → **DONE** — full error table with Time/Level/Source/Message/Log columns, parsed via `_parse_log_errors()`, sidebar error cards
55. ~~Log-level filter buttons~~ → **DONE** — CRITICAL/ERROR/WARNING chips with counts, color-coded
56. WCAG AA contrast pass — **NEEDS AUDIT** — multiple low-opacity text values (`/50`, `/60`, `/70`) likely fail 4.5:1 ratio
57. ~~Stable machine-readable error codes~~ → **DONE at API level** — `ErrorCode` Literal with 13 codes (`INSUFFICIENT_MEMORY`, `PLACEMENT_FAILED`, etc.), `ApiError` class, `ErrorResponse` envelope; dashboard consumption still minimal
58. ~~App shell: surface stderr + exit reason~~ → **DONE** — `RunnerFailed` with `error_message` + `diagnostics` (MetalGpuTimeout, RingTransportError, CudaOom, etc.); dashboard surfaces in instance cards

### Diagnosed — Needs Fix (from 2026-09-02/03 investigation)
62. Smox node isolation — API server down on exo-amd container (10.2.0.76); zenoh discovers but HTTP unreachable. Needs container restart + code update from `main` to `fix-memory-error` — **DEFERRED (infra-only, no code path in this branch)**
~~63. Download queue stall~~ → **DONE (`8bed4ca4` T53)** — `DownloadStalled` type + `_stall_watchdog()` in DownloadCoordinator polls active downloads, cancels stalled ones after `EXO_DOWNLOAD_STALL_TIMEOUT_SECS` (30min default); `_stalled_models` set distinguishes watchdog cancel (preserves status for retry) from user cancel (clears status)
64. Election cycling breaks multi-step flows — download/placement state lives on master in-memory; master change loses in-flight operations. Needs persistent download state or master-pin protocol — **DEFERRED (P5 architectural, tracked in research/master-implementation-plan)**
~~65. Network traffic display~~ → **BACKEND DONE (`67020f5f`)** + **UI DONE (`ea766091`)** — per-node rx/tx in placement panel
~~66. Bandwidth-aware pipeline placement~~ → **duplicate of P1 #35, DONE (`35153898`)**
~~67. TP single-node heuristic~~ → **duplicate of P1 #37, DONE (`df7f0aa9`)**
~~68. Pipeline activation memory estimate~~ → **duplicate of P1 #38, DONE (`c4165eee`)**

### Open Questions
59. Commit + push CUDA memory limit fix upstream? (genuine bug fix) — **DECISION: DEFER** — fix is in our `fix-memory-error` branch; upstream PR requires maintainer buy-in and a clean isolated commit. Track as follow-up when branch merges.
~~60. Wire `EXO_ZENOH_NAMESPACE` env var to `--namespace` CLI?~~ → **DONE (`1c3af9bf` T54)** — `default=os.getenv('EXO_ZENOH_NAMESPACE', __version__)` at `main.py:595`; env var and CLI now agree; `--help` documents env var
61. Upstream PR: CUDA ring attention `_is_cuda_backend()` stream fix — **DECISION: DEFER** — function at `ring_attention.py:58` correctly checks `linux + gpu`; our local fix is sufficient. Upstream PR only needed if ml-explore/mlx adds CUDA backend support.
