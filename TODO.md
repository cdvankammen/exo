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
29. Guard `publish_bytes` against malformed events (malformed event kills process → exit 1)
30. Pre-prefill eviction in `cache.py` (OOM during prefill, #2182)
31. Periodic cache cleanup + gc.collect() (idle memory leak, #2262)
32. Input length limiter (66k-token OOM, #560)
33. Node identity periodic re-announce (lost startup announcement → invisible node)
34. ~~Log rotation for runner stdout/stderr (Linux log grew to 14GB → disk full)~~ → **DONE (`57eab048`)** — runner subprocess logs rotate; complements main `exo.log` rotation (`009b43c6`)

### P1 — Quick Wins (week 1-2)
35. Bandwidth-aware pipeline placement (#957 water-filling algorithm)
36. Ring context configurable (`EXO_RING_ADMISSION_CONTEXT` env var)
37. TP single-node heuristic (don't split if one node fits)
38. Pipeline activation memory estimate (~10% of weights)
39. OOM graceful degradation (evict → halve batch → retry instead of SIGABRT)
40. Auto-restart VRAM check (skip restart if memory >90%)
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
53. Logs page: per-node + aggregated "all nodes" view
54. Error table on Logs page (structured WARNING/ERROR list)
55. Log-level filter buttons
56. WCAG AA contrast pass
57. Stable machine-readable error codes (`INSUFFICIENT_MEMORY` etc.)
58. App shell: surface stderr + exit reason (don't discard)

### Diagnosed — Needs Fix (from 2026-09-02/03 investigation)
62. Smox node isolation — API server down on exo-amd container (10.2.0.76); zenoh discovers but HTTP unreachable. Needs container restart + code update from `main` to `fix-memory-error`
63. Download queue stall — bad model card blocks entire queue; needs per-download timeout + dead-letter handling + "stalled" dashboard warning
64. Election cycling breaks multi-step flows — download/placement state lives on master in-memory; master change loses in-flight operations. Needs persistent download state or master-pin protocol
65. Network traffic display — show live bytes in/out per node in dashboard topology (backend DONE `67020f5f`, UI IN PROGRESS)
66. Bandwidth-aware pipeline placement — use topology bandwidth_mbps data for water-filling layer allocation across pipeline stages (P1 #35)
67. TP single-node heuristic — don't tensor-shard when one node has enough memory for the full model
68. Pipeline activation memory estimate — reserve ~10% of weight memory for activations

### Open Questions
59. Commit + push CUDA memory limit fix upstream? (genuine bug fix)
60. Wire `EXO_ZENOH_NAMESPACE` env var to `--namespace` CLI?
61. Upstream PR: CUDA ring attention `_is_cuda_backend()` stream fix
