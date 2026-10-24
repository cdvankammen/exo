# Research: Memory Bandwidth & SSD-as-RAM PRs on exo-explore/exo

> **Date:** 2026-08-05
> **Author:** GitHub Copilot agent (session research)
> **Status:** active
> **Tags:** memory, bandwidth, ssd, kv-cache, offload, tensor-parallelism, research

## Purpose

Survey of open/closed pull requests on `exo-explore/exo` related to:
1. **More memory bandwidth** (interconnect, Thunderbolt, RDMA)
2. **Using SSD/disk as additional RAM** (offload, persistence, external storage)
3. **Loading models larger than a single node's RAM** (partitioning, KV cache compression, GPU VRAM)

Search was fuzzy — anything *slightly* related was included. Full PR list captured 2026-08-05.

---

## Tier 1 — Disk/SSD as Memory (KV cache offload & external storage)

| # | State | Title | Notes |
|---|-------|-------|-------|
| **#1830** | closed | KV cache: per-conversation slot separation + disk persistence/offload | **Closest thing to "SSD as RAM" in the codebase.** Offloads conversation KV state to disk, freeing RAM for weights. Last updated 2026-08-01 — discussion still active despite closed state. |
| **#2159** | closed | Add opt-in disk persistence for the KV prefix cache | Disk-backed KV prefix cache; reload prompt cache from SSD. Updated 2026-08-01. |
| **#1820** | closed | KV cache: multi-conversation slot separation + disk persistence | Earlier iteration of #1830. |
| **#1765** | closed | Rework model storage directory management (for external storage) | Point exo at external drives for model storage. |
| **#1675** | open | Ciaran/storage management | Likely successor to #1765; ongoing. |
| **#1598** | open | feat: auto-discover models from external directories | Run models from Thunderbolt/external SSDs. |
| **#1574** | closed | worker: add EXO_MODELS_PATH for pre-downloaded model directories | Point workers at any dir (incl. external drives). |

## Tier 2 — Memory Bandwidth & Interconnect

| # | State | Title | Notes |
|---|-------|-------|-------|
| **#2229** | open | perf: faster pipeline decode: TPS 10%-40% and Layer loading 60%+ (token relay + bandwidth-aware placement) | Big bandwidth win: 60%+ faster layer loading over interconnect. |
| **#2160** | open | feat: stripe RDMA traffic across multiple Thunderbolt cables (multi-rail jaccl) | **Bonds multiple TB cables** for aggregated bandwidth. |
| **#2195** | open | feat(placement): bond parallel RDMA links in JACCL device matrix | Parallel RDMA link bonding. |
| **#1088** | closed | feat: implement bandwidth-aware shard assignment for pipeline parallelism | Places shards by actual link bandwidth. |

## Tier 3 — Fitting Larger Models in Available RAM

| # | State | Title | Notes |
|---|-------|-------|-------|
| **#2255** | open | feat: sequence-parallel Ring Attention for MLX | Splits long sequences across devices; run contexts that don't fit one node. |
| **#2064** | open | Asymmetric tensor parallelism for uneven-memory clusters | 64GB + 16GB Macs both contribute proportionally. |
| **#1821** | open | feat: asymmetric tensor parallelism for heterogeneous clusters | Same concept. |
| **#2226** | open | add flops + memory weighted partitioning strategy | Smarter model-to-node mapping by memory capacity. |
| **#2227** | open | fix: capacity-aware proportional layer allocation | Layers allocated proportional to node memory. |
| **#2253** | open | feat: support manual pipeline layer allocation | Manual control over layer placement. |
| **#2251** | open | fix: reserve memory headroom before prefill | Prevents OOM during prefill. |
| **#2240** | open | Add configurable context-window with memory-safety validation | Trade context window for model size. |
| **#2252** | open | fix: prefer accelerators in heterogeneous placement | Prefer GPU/ANE nodes for memory-heavy shards. |
| **#1626** | open | Return error chunk in low memory situations | Graceful degradation instead of crash. |
| **#1627** | open | OOM settings | Configurable OOM behavior. |
| **#996** | closed | feat: add memory pressure awareness to placement decisions | Placement considers real-time memory pressure. |
| **#994** | closed | Add differentiation between available and total memory | Uses *available* memory, not total. |
| **#1719** | closed | feat: add --memory-fraction CLI arg to cap available memory for placement | Reserve system memory while loading large models. |
| **#1441** | closed | feat: add intermediate model-fit state for cluster-capacity-only models | Shows models that fit the *cluster* but not a single node. |

## Tier 4 — KV Cache Compression (Free RAM for Bigger Models)

| # | State | Title | Notes |
|---|-------|-------|-------|
| **#2242** | open | Expose existing KV-cache quantization behind an env-var opt-in | Quantized KV cache = less RAM per conversation. |
| **#1988** | open | feat: EXO_KV_CACHE_BITS env var + step=16384 for QuantizedKVCache | Configurable KV cache quantization bits. |
| **#1860** | open | Fix unbounded KV cache memory growth by wiring up RotatingKVCache | Prevents KV cache eating all RAM. |
| **#1990** | open | fix: skip KV cache quantization in single-node BatchGenerator mode | Bug fix for KV cache quantization. |
| **#1886** | closed | Reduce memory consumption by adding Flash Attention to Qwen3.5 and Gemma 4 | Flash attention = dramatically less memory per token. |
| **#1127** | open | Support for FLASH | Flash attention support broadly. |

## Tier 5 — GPU VRAM as Additional Memory (Non-Mac Path)

| # | State | Title | Notes |
|---|-------|-------|-------|
| **#2216** | open | feat: report CUDA GPU VRAM as node memory on Linux | Cluster sees GPU memory alongside system RAM. |
| **#2103** | open | feat: Add Native CUDA Support for Linux GPU Inference | Native CUDA inference. |
| **#1284** | open | feat: Add PyTorch backend for NVIDIA GPU support | PyTorch backend = access to NVIDIA VRAM. |
| **#1660** | open | feat(Linux): Tinygrad runner for LLM inference on Linux | Tinygrad runner with GPU memory support. |

## Tier 6 — Speculative Decoding (More Output Per Memory Access)

| # | State | Title | Notes |
|---|-------|-------|-------|
| **#2079** | open | Drafter abstraction + multi-device coupled-drafter speculative decoding | Small model drafts, large verifies; more throughput per memory access. |
| **#2110** | open | Add native MTP for Qwen3.6 MLX models | Multi-token prediction = multiple tokens per memory read. |

---

## Key Takeaways

1. **#1830 / #2159** (KV cache disk offload) are the closest to "SSD as RAM" — closed but actively discussed (last updated 2026-08-01). Likely being reworked or merged into a different approach.
2. **#2160 / #2195** (multi-rail Thunderbolt bonding) aggregate bandwidth across cables.
3. **#2064 / #1821** (asymmetric tensor parallelism) let uneven nodes pool memory for bigger models.
4. **#2255** (Ring Attention) splits long-context memory across nodes.
5. **#1675 / #1598** (external storage management) run models from external drives.

**Watch list:** #1830, #2159, #1675, #2255, #2160, #2195, #2064, #1821, #2229.

---

## Investigation: Why #1830 and #2159 Were Both Closed (2026-08-05)

### #1830 (closed Jun 9, 2026)
- **Author:** aidiffuser
- **Closure reason (author comment):** *"Closing for now — reworking this KV-cache disk-persistence feature onto the current Zenoh-based main and validating it across model architectures. Will resubmit a fresh PR once it's ready."*
- **Verdict:** Explicitly abandoned. Code was written against pre-zenoh architecture. Porting it would fight the new codebase structure. **Do NOT port.**
- **Files changed:** `cache.py`, `runner.py`, `generate.py`, `batch_generate.py` (all under old paths — runner was at `worker/runner/llm_inference/runner.py`, now at `worker/runner/runner.py`).

### #2159 (closed ~Aug 1, 2026) — Explicitly supersedes #1830
- **Author:** aidiffuser (same author)
- **Files changed (8):** `builder.py`, `cache.py`, `disaggregated/serve.py`, `batch_generate.py`, `generate.py`, `runner.py` (correct new path), plus 2 new test files.
- **Key design improvements over #1830:**
  - Opt-in (`EXO_KV_DISK_PERSISTENCE=1`) instead of opt-out
  - Uses `EXO_CACHE_HOME` (XDG paths) instead of hardcoded `~/.exo`
  - Global size cap across ALL models (not per-model)
  - Handles non-standard cache types: DeepseekV4Cache branch state with None/int/list values, zero-width arrays, ArraysCache with None entries
  - Atomic writes: tmp + rename, tokens written last so slot only discoverable when complete
  - Janitor TTL-sweeps all model directories at init and after every flush (cross-model LRU)
  - Failsafe: any load problem logs warning and falls back to recompute
  - Compatible with stock `mlx_lm.save_prompt_cache` / `load_prompt_cache`
  - 9 unit tests (round-trip, mlx-lm format compat, non-trimmable guard, prefix capping, opt-in default)
- **Closure reason:** Unknown — branch deleted, no explanation. Possible: maintainer asked for changes offline, conflicts with in-flight work, waiting for #2158 (trim_cache V4 fix).
- **Dependency note:** References PR #2158 (fix trim_cache bug for V4 entries) as independent but related.

### Current Codebase Status (verified 2026-08-05)
- **Zero KV disk persistence code exists in main or fix-memory-error branch.**
- No `EXO_KV_DISK_PERSISTENCE`, no `save_prompt_cache`, no `load_prompt_cache` in `src/exo/`.
- The feature is genuinely absent — neither PR landed.
- `git log main --grep="disk" --grep="kv.*persist"` shows nothing related merged.

### Recommendation for Merge Campaign
- **Skip #1830 entirely** — stale, abandoned, wrong architecture.
- **#2159 is the correct candidate** but was also closed without explanation. Need to:
  1. Read the full #2159 diff (1194 lines, 8 files)
  2. Verify it applies cleanly against current codebase (paths match, `builder.py` exists, etc.)
  3. Check if #2158 (trim_cache fix dependency) is already in main
  4. Attempt port; flag for user decision if conflicts arise.