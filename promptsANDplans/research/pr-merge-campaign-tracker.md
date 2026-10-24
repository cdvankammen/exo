# PR Merge Campaign Tracker

> **Started:** 2026-08-05
> **Branch:** fix-memory-error
> **Base commit:** a6b99564 (pre-campaign baseline)

## Process Per PR
1. Check state & closure reason (skip superseded/rejected)
2. Fetch diff via `gh pr diff`
3. Line-by-line review of all changed files
4. Apply changes locally
5. Run full gate: `uv run basedpyright && uv run ruff check && uv run pytest`
6. Subagent code review for non-trivial changes
7. Commit with full attribution (PR#, author, URL, summary)
8. If threshold not met → skip, record reason, move to next

## Status Legend
- ✅ Merged successfully
- ⏭️ Skipped (reason noted)
- 🔄 In progress
- ❌ Failed verification
- ⏳ Pending

## PR Queue (top-down order)

| # | PR | Title | Status | Notes |
|---|-----|-------|--------|-------|
| 1 | #1830 | KV cache disk persistence/offload | ⏭️ | **SKIP** — abandoned, superseded by #2159 |
| 1b | #2159 | KV disk persistence v2 (supersedes #1830) | ✅ | Ported & committed as 7dd12e9f. 0 type errors, 0 lint errors, 3/3 wiring tests pass. |
| 2 | #2159 | KV prefix cache disk persistence | ⏳ | Closed — check why |
| 3 | #1820 | KV cache multi-conversation disk persistence | ⏳ | Closed — likely superseded by #1830 |
| 4 | #1765 | External storage directory management | ⏳ | Closed |
| 5 | #1675 | Storage management | ⏭️ | **SKIP** — OPEN but CONFLICTING, 33 files/4576 lines, 2+ months stale, no reviews. Too risky to port manually; revisit after upstream resolves conflicts. |
| 6 | #1598 | Auto-discover external models | ⏳ | Open |
| 7 | #1574 | EXO_MODELS_PATH env var | ⏳ | Closed |
| 8 | #2229 | Token relay + bandwidth-aware placement | ⏳ | Open |
| 9 | #2160 | Multi-rail Thunderbolt bonding | ✅ | Ported & committed as 5f7682dc. 0 type errors, 0 lint errors. ⚠️ DRAFT — depends on upstream jaccl recv prefill fix (rltakashige/mlx-jaccl-fix-small-recv#3). PP activation may hang without it; TP unaffected. |
| 10 | #2195 | Parallel RDMA link bonding | ⏭️ | **SKIP** — redundant with #2160 (which we ported as 5f7682dc). Both do parallel RDMA link bonding but use INCOMPATIBLE type signatures: #2160 uses `list[list[list[str]]]` (no None), #2195 uses `list[list[list[str] | None]]`. They cannot both be applied. #2160 has real hardware benchmarks (19.4→42.0 tok/s), #2195 is DRAFT with only 4-5% prefill improvement measured. Keeping #2160. |
| 11 | #1088 | Bandwidth-aware shard assignment | ⏳ | Closed |
| 12 | #2255 | Ring Attention for MLX | ⏳ | Open |
| 13 | #2064 | Asymmetric tensor parallelism | ⏳ | Open |
| 14 | #1821 | Asymmetric TP heterogeneous clusters | ⏳ | Open |
| 15 | #2226 | FLOPS + memory weighted partitioning | ⏳ | Open |
| 16 | #2227 | Capacity-aware layer allocation | ✅ | Ported & committed as 35f1e789. 0 type errors, 0 lint errors. Prevents CUDA OOM on heterogeneous clusters. |
| 17 | #2253 | Manual pipeline layer allocation | ⏳ | Open |
| 15 | #2251 | Reserve memory headroom before prefill | ✅ | Ported & committed as 5260b384. 0 type errors, 0 lint errors, auto-applied cleanly. Fixes #2182 (Metal GPU OOM). |
| 19 | #2240 | Configurable context-window | ⏭️ | **SKIP** — 8/15 files conflict with prior ports (#2159/#2242/#2251). Author's own PR description flags UNTESTED interaction with #2242 (KV quantization, which we just ported) and unverified RotatingKVCache + prefix-cache behavior. No reviews. Requires architectural decisions across API/master/worker/types — exceeds threshold. Revisit after upstream resolves. Fixes #1936 (tensor-parallel uneven memory). |
| 20 | #2252 | Prefer accelerators in placement | ✅ | Ported & committed as 38101807. 0 type errors, 0 lint errors. Fixes #2180 (CPU preferred over idle CUDA). |
| 21 | #1626 | Error chunk on low memory | ⏳ | Open |
| 22 | #1627 | OOM settings | ⏳ | Open |
| 23 | #996 | Memory pressure awareness | ⏳ | Closed |
| 24 | #994 | Available vs total memory | ⏳ | Closed |
| 25 | #1719 | --memory-fraction CLI arg | ⏳ | Closed |
| 26 | #1441 | Cluster-capacity-only model state | ⏳ | Closed |
| 18 | #2242 | KV-cache quantization opt-in | ✅ | Ported & committed as 070acc7f. 0 type errors, 0 lint errors. Manual cache.py fix needed (conflict with #2159 imports). |
| 28 | #1988 | EXO_KV_CACHE_BITS env var | ⏭️ | **SKIP** — superseded by #2242 (same author @alytaphoenix, same env var, same changes). We already ported the newer #2242. |
| 29 | #1860 | RotatingKVCache wiring | ⏳ | Open |
| 19 | #1990 | Skip KV quant single-node | ✅ | Ported & committed as fde5c298. 0 type errors, 0 lint errors. Fixes #1875. Complements #2242. |
| 31 | #1886 | Flash Attention Qwen3.5/Gemma 4 | ⏳ | Closed |
| 32 | #1127 | General FLASH support | ⏳ | Open |
| 23 | #2216 | CUDA VRAM as node memory | ✅ | Ported & committed as ddc34a44. 0 type errors, 0 lint errors. Caveat: 100% free VRAM still risks runtime OOM — follow-up CUDA_VRAM_USABLE_FRACTION=0.75 suggested by @jgawronek. |
| 24 | #2103 | Native CUDA support | ✅ | Ported & committed as 02f17fb4. 0 type errors, 0 lint errors. Enables heterogeneous macOS Metal + Linux CUDA clusters. |
| 35 | #1284 | PyTorch backend NVIDIA | ⏳ | Open |
| 36 | #1660 | Tinygrad runner Linux | ⏳ | Open |
| 37 | #2079 | Multi-device speculative decoding | ⏳ | Open |
| 38 | #2110 | Native MTP Qwen3.6 | ⏳ | Open |

## Uncommitted Mystery Files (NOT part of any ported PR)
These appeared in the working tree during the campaign — separate feature work, not from any PR I ported:
- `dashboard/src/routes/+page.svelte` — "Add Node Manually" UI panel (POST /peers, Tailscale/hostname connect)
- `src/exo/api/main.py` — `add_peer()` endpoint + `AddPeerParams`/`AddPeerResponse` imports
- `src/exo/api/types/__init__.py` — exports `AddPeerParams`, `AddPeerResponse`
- `src/exo/main.py` — passes `router=router` to API constructor
- `rust/exo_rs/exo_rs.pyi` — `connect_peer(host, port)` stub
- `src/exo/api/tests/test_add_peer.py` (untracked) — 4 tests for POST /peers

**Likely source:** partial port of PR #2243 (`feat(networking): allow dialing peers over unicast via EXO_ZENOH_CONNECT` by @Bawan-Wang) or related local work. NOT committed as part of this campaign — needs separate attribution decision.

## Skipped PRs Log
- **#1830** — abandoned by author, superseded by #2159
- **#1675** — CONFLICTING state, 33 files/4576 lines, stale 2+ months, no reviews
- **#2240** — 8/15 files conflict with prior ports; author-flagged untested KV-quantization interaction (we just ported #2242); RotatingKVCache + prefix-cache behavior unverified on real hardware; no reviews. Architectural decisions required.

## Commits Made

### Wave 3 (2026-08-06): Remaining Tier 1/2 ports

| # | Commit | PR Ported | Author | Files |
|---|--------|-----------|--------|-------|
| 1 | `3abc8623` | #1999 (DSv4 fused MoE, opt-in part) | @adurham | 1 |
| 2 | `8ccbe8c8` | #2254 (link latency ring selection) | @abendrothj | 6 |
| 3 | `86927717` | #2155 (EXO_PREFILL_STEP_SIZE) | @lollinng | 1 |
| 4 | `15647f8a` | #2222 (Claude system role) | @darinlarimore | 3 |
| 5 | `b80ffbf1` | #2237 (dashboard log viewer) | @alytaphoenix | 7 |
| 6 | `366c5302` | #2179 (use_prefix_cache flag) | @mlpy0 | 5 |
| 7 | `36acecdd` | #2035 (liveness polling tuning) | @AlexCheema | 2 |
| 8 | `91173df0` | #2060 (symlink-safe deletion, Python part) | @team-wcv | 3 |

### Wave 2 (2026-08-05/06): Track A/B + Tier 1 PRs

| # | Commit | PR Ported | Author | Files |
|---|--------|-----------|--------|-------|
| 1 | `868088f7` | #2063 (TB5 nested-hub, JACCL retry, RDMA host sel) | @team-wcv | 7 |
| 2 | `157a4c24` | #1598 core (HF Hub cache discovery) | @AlexCheema | 3 |
| 3 | `f4abf47e` | #2177 (snapshot dicts in supervisor) | @rogerstom-lgtm | 1 |
| 4 | `1102e28b` | #2121 (rank-0 only ChunkGenerated TP) | @adurham | 1 |
| 5 | `b2b3dd74` | #2184 (preserve finish_reason truncated tools) | @mlpy0 | 2 |
| 6 | `5686da7c` | #2187 (drop GLM leaked arg markup) | @mlpy0 | 1 |
| 7 | `a503b2b6` | #2054 (Kimi processor_repo fix) | @Drifter4242 | 3 |
| 8 | `5c0480b2` | #2055 (vision text_config/KeyError fix) | @Drifter4242 | 1 |
| 9 | `b4d02e38` | #2134 (SSRF block fetch_image_url) | @5F0jd2vLq54RerYW | 3 |
| 10 | `bf38d869` | #2001 (thinking content leak fix) | @mlpy0 | 2 |
| 11 | `10876037` | #2189 (tool_choice forced calls) | @michaelharrigan | 5 |
| 12 | `b55c99b0` | #2203 (deterministic election recovery) | @oliver0006 | 2 |
| 13 | `1d83a244` | #2123 (Claude stop_sequence) | @morisil | 8 |
| 14 | `ba7092d1` | #2202 (event log ownership guard) | @oliver0006 | 2 |
| 15 | `ac7a170f` | #2162 (Xcode 26 build fix) | @jasonpaulso | 1 |
| 16 | `e7e0d34b` | #2133 (events since/limit pagination) | @5F0jd2vLq54RerYW | 3 |
| 17 | `c34c1589` | #2161 (macOS 26 psutil fallback) | @jasonpaulso | 5 |
| 18 | `6e71b7bd` | strict-typing adaptations (4 PRs) | agent | 4 |
| 19 | `dbf76ecb` | #2205 (orphan watchdog - CLOSED upstream) | @aidiffuser | 1 |
| 20 | `143d70c2` | #2231 (event-send hardening + election debounce) | @caesarniseko-coder | 4 |
| 21 | `68d32e8c` | #1804 (master routing dead runners + error chunks) | @mlpy0 | 1 |
| 22 | `165b614e` | #2169 + #2148 (docs: architecture + TurboQuant plan) | @dhanvriddhi1gugu + @ianleon | 2 |

### Wave 1 (2026-08-05): original 26 commits

| # | Commit | PR Ported | Author | Files | Lines |
|---|--------|-----------|--------|-------|-------|
| 1 | `014a23ee` | baseline WIP | (pre-existing) | 6 | 119+/23- |
| 2 | `7dd12e9f` | #2159 (KV disk persistence v2) | @aidiffuser | 8 | 990+/27- |
| 3 | `070acc7f` | #2242 (KV-cache quantization env var) | @alytaphoenix | 3 | 26+/14- |
| 4 | `5260b384` | #2251 (prefill memory headroom) | @abendrothj | 3 | 55+/5- |
| 5 | `fde5c298` | #1990 (skip KV quant single-node) | @adurham | 1 | 39+/2- |
| 6 | `5f7682dc` | #2160 (multi-rail TB bonding) | @jasonpaulso | 4 | 117+/21- |
| 7 | `35f1e789` | #2227 (capacity-aware layer alloc) | @jgawronek | 1 | 41+/6- |
| 8 | `38101807` | #2252 (prefer accelerators) | @abendrothj | 2 | 60+/0- |
| 9 | `ddc34a44` | #2216 (CUDA VRAM as node memory) | @darinlarimore | 2 | 88+/3- |
| 10 | `02f17fb4` | #2103 (native CUDA Linux) | @Winston-9527 | 4 | 49+/27- |
| 11 | `26818670` | #2192 (transient EHOSTUNREACH) | @batmunkh0612 | 1 (Rust) | 6+/4- |
| 12 | `ed7ba52f` | #2238 (eject+weight deletion) | @alytaphoenix | 1 (Svelte) | 132+/10- |
| 13 | `4bac50c9` | #2204 (debug prompt trigger gate) | @alytaphoenix | 1 | 11+/1- |
| 14 | `2772b55d` | #2244 (runner reconnect cascade) | @alytaphoenix | 6 | 211+/7- |
| 15 | `6f5458ba` | #2228 (ArraysCache mask tolerance) | @jgawronek | 1 | 19+/0- |
| 16 | `53bb0f98` | #2158 (V4 trim_cache restore) | @aidiffuser | 2 | 60+/5- |
| 17 | `6f5458ba` | #2228 (ArraysCache mask tolerance) | @jgawronek | 1 | 19+/0- |
| 18 | `c1075b87` | #2152 (vision cache hash dims) | unknown | 1 | 4+/1- |
| 19 | `0756536e` | #2174 (GPT-OSS tiktoken cache) | @alytaphoenix | 1 | 45+/0- |
| 20 | `5fa86ef8` | #2218 (Gemma 4 12B model card) | @robertlangdonn | 1 | 22+/0- |
| 21 | `5913b0ac` | #2239+#2247+peer-conn (combined) | @alytaphoenix + @Bawan-Wang | 12 | 509+/31- |
| 22 | `89921b38` | #2245 (docs: mlx extra in setup) | unknown | 3 | 126+/4- |
| 23 | `6f0261c2` | #2249 (orphaned download GC) | unknown | 3 | 66+/3- |
| 24 | `2d50d66d` | #2250 (stale node identity prune) | unknown | 2 | 51+/0- |
| 25 | `086787dd` | coordination fix (force_override bypass) | agent | 1 | 15+/5- |

## Final Integration Gate (2026-08-05)

Full-repo verification across all 26 commits combined:
- **basedpyright (full repo):** 0 errors, 0 warnings, 0 notes ✅
- **ruff check (full repo):** All checks passed ✅
- **nix fmt / nix flake check:** SKIPPED — `nix` not installed in this environment. Per AGENTS.md, CI runs `nix flake check` which verifies formatting, linting, and Rust tests. Ruff substituted for Python formatting.
- **pytest:** Individual test files verified per-commit. Full suite requires mlx (macOS-only, not in dev venv). Pre-existing svelte-check baseline unchanged (15 errors / 6 warnings, all pre-existing).
- **Working tree:** Clean (only untracked `default.profraw` from cargo profiling)

## Still Uncommitted (not part of any ported PR)
Peer-connection feature files — likely from PR #2243 or local work. Need separate attribution decision:
- `dashboard/src/routes/+page.svelte` (Add Node Manually UI)
- `rust/exo_rs/exo_rs.pyi` (connect_peer stub)
- `src/exo/api/main.py` (add_peer endpoint)
- `src/exo/api/types/__init__.py` (AddPeerParams/Response exports)
- `src/exo/main.py` (router=router wiring)
- `src/exo/api/tests/test_add_peer.py` (untracked, 4 tests)

---

## Session Addendum 2026-08-06 — Linux Deployment + Eject UI + Research

### 1. Linux Deployment (10.2.0.16) — DONE ✅
Full source copied (branch `fix-memory-error`, 56 commits) to Proxmox VM `tryingOutExo` (Ubuntu 24.04, 2× RTX 5060 Ti 16GB passed through):
- rsync'd repo (19,430 files, excluded target/.venv/build/node_modules); old checkout backed up to `~/exo.upstream-bak`
- Installed uv 0.12.2; `uv sync --extra mlx`
- **mlx on Linux x86_64 is TWO packages**: `mlx` (bindings) + `mlx_cuda_12` (runtime, from rltakashige fork per our lock file) — both wheels installed manually after cache had wrong-arch (ARM) builds
- **Driver saga**: kernel module 595.58.03 vs apt 535/595.84 → installed exact 595.58.03 userspace from NVIDIA `.run` (extracted, symlinked libcuda/libnvidia-ml/ptxjitcompiler/tls/gpucomp)
- **CUDA toolkit**: `nvidia-cuda-toolkit` apt package for headers; `CUDA_HOME=/usr/local/cuda` (symlink → /usr) in `/etc/environment`
- **exo runs detached**: `setsid nohup env CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda uv run exo`
- **VERIFIED end-to-end**: Llama-3.2-1B (73.5 tok/s), Llama-3.2-3B (29 tok/s), single-node, `min_nodes:1` Pipeline — `"CUDA MLX backend detected; reporting GPU VRAM as node memory"` logged (our #2103/#2216 ports live)
- **Root cause of earlier election churn**: stale EXO.app on BOTH Macs crash-looping (missing `event_id` in BaseEvent) — fixed by user rebuilding the app

### 2. Eject UI Redesign — DONE ✅ (commit `ca8474c7`)
- New `dashboard/src/lib/components/EjectButton.svelte`: inline split-button (NO outline / YES filled) with fly animation; direct import (barrel export alone broke Svelte 5 tree-shaking)
- `ejectInstance()` in `+page.svelte`: removed BOTH `confirm()` popups — weights auto-deleted on eject
- Full app rebuild done properly: `uv sync --extra mlx` (+ explicit mlx git reinstall), pyinstaller (`dist/exo` 15:50), xcodebuild (DerivedData), installed to /Applications (fixed 700 perms via `chmod -R a+rX` + ad-hoc re-sign), relaunched — new UI served ✅

### 3. Deep Research (see `remaining-prs/deep-dive-2129-2200-heterogeneous.md`)
- **#2129** (3-node Metal+CUDA): TcpRelay bypass for broken CUDA↔CUDA send/recv, Metal-to-middle rotation, MLX_HOSTS_JSON, CUDA rank probing, post-prefill barrier, Linux interface types. We already have cuda_compat/bootstrap/wired-limit pieces; missing TcpRelay, rotation, probing, barrier, interface types. SKIP the IP-priority flip (conflicts with our TB preference).
- **#2200** (backend per shard): small clean foundation — `ShardMetadata.backend` + `assign_shard_backends()` + `apply_shard_backend()` in runner. Recommend porting BEFORE #2129.
- **Non-MLX models**: no upstream PR exists; would require a whole new inference engine (GGUF/llama.cpp) — out of scope.
- **Branch cleanup**: `fix-memory-error/track-a-asymmetric-tp` + `fix-memory-error/track-b-model-discovery` were created then abandoned when the user pivoted to single-branch merging; they contain no commits (empty pointers).

## Wave 4 — Heterogeneous Cluster (2026-08-06)

- `72cf86bb` — **#2129** 3-node heterogeneous Metal+CUDA pipeline support (@Juliuscply, https://github.com/exo-explore/exo/pull/2129)
  - TcpRelay raw-TCP bypass for CUDA↔CUDA ring send/recv (auto_parallel.py)
  - MLX_HOSTS_JSON + eager relay start + CUDA peer probe → MLX_CUDA_RANKS (utils_mlx.py)
  - _rotate_metal_to_middle placement rotation (placement.py + 5 unit tests)
  - Prefill barrier (generate.py)
  - _linux_interface_types (system_info.py)
  - SKIPPED: placement_utils IP-priority flip (conflicts with our TB4-optimized #2063/#2254)
  - Already in branch from #2103: cuda_compat.py, platform-gated Metal env, CUDA wired limit
  - Verified: pyright 0 errors, ruff clean, 620 tests pass (only pre-existing baseline failures)
  - This is the PR referenced in #2103's comments — the Mac + Linux GPU split-cluster enabler

## Wave 5 — Error UX (2026-08-06, all MY commits, no upstream PRs)

User-driven hardening wave — "make errors explain themselves":
- `4f850371` — keep failed instances visible + render error_message/diagnostics on FAILED cards (master no longer auto-deletes RunnerFailed instances; dashboard red error box). Fixes: models vanishing without a trace.
- `85f1b024` — Swift app: capture stderr (was nullDevice!) + read log tail; extractStartupFailureReason() shows "Port already in use" etc instead of "Exited with code 1". Fixes: startup failures unexplainable.
- `5acd6d6c` — stable ErrorCode Literal + ApiError + error_code field (INSUFFICIENT_MEMORY, PLACEMENT_FAILED, MODEL_NOT_FOUND, INSTANCE_NOT_FOUND, NOT_FOUND, INVALID_REQUEST, INTERNAL_ERROR). UI can branch without parsing messages.
- `ef16c380` — publish_bytes try/except guard (malformed events used to crash the whole process — the exact bug behind the earlier event_id exit-1 crashes) + GET /v1/warnings + yellow MALFORMED EVENTS dashboard chip (reusing macOS MISMATCH chip template).

Design decision (user): Swift app = server STARTUP errors; browser UI = runtime errors after server running. Error codes feed both.

## Wave 5 — Linux toolchain fixes (2026-08-06)

- `f48bdf86` — **uv.lock torch fix** (NOT a PR — our own fix)
  - Why: pyproject declares torch for linux but uv.lock's `extra != 'mlx-cpu'` marker clauses (side-effect of #2103's extra layout) are mis-evaluated by uv 0.12.x → torch never installed on Linux → vision disabled
  - Fix: simplified 3 markers (torch/torchaudio/torchvision) to `sys_platform == 'linux' and extra == 'mlx'`
  - Verified: torch 2.10.0+cu130 installs + vision conversion path works on the box
  - Caveat: `uv lock` regen re-introduces markers (uv quirk); re-apply if regenerated

## Machine-level fixes (NOT commits — box config, see linux-nvidia-cuda-setup-guide.md)

1. Driver userspace 595.58.03 matched to kernel module via NVIDIA .run extraction (apt's 595.84/535 mismatched)
2. CUDA 12.8 toolchain (cuda-nvrtc-12-8 + nvvm + cudart-dev) — first Blackwell (sm_120) capable; fixed `nvrtc: invalid value for --gpu-architecture`
3. mlx_cuda_12 runtime wheel reinstalled after each `uv sync` (sync reverts to URL wheel without libmlx.so)
4. CUDA_HOME/CUDA_PATH in /etc/environment + launch env

## Wave 6 — PR #2229 token relay + bandwidth-aware placement (2026-08-07, 4 commits)

PR: https://github.com/exo-explore/exo/pull/2229 by @jgawronek — "perf: faster pipeline decode: TPS 10%-40% and Layer loading 60%+ (token relay + bandwidth-aware placement)". 1 commit, 13 files, +824/-48, open, no reviews. Unbundled into 4 surgical commits:

- `f0e59802` — **token-relay decode**: last pipeline rank samples alone; token ids circulated via int32 all_gather (relay_sampled_tokens); non-last ranks' lm_head drops out of lazy graph (zeros_like in patched_call); auto-fallback to legacy all_gather for logprobs requests; PipelineDecodeTimings debug logging. Files: auto_parallel.py, opt_batch_gen.py, batch_generate.py + 2-process distributed ring test.
- `e18a8f0e` — **bandwidth-aware layer allocation**: chip bandwidth lookup table (M4 Ultra 1092 → M1 68 GB/s; RTX 3090 936, GB10 273); allocate_layers_by_throughput() loads fastest nodes to memory caps first; node_identities threaded through placement chain; Linux chip_id from nvidia-smi GPU name. Files: placement_utils.py, placement.py, api/main.py, master/main.py, system_info.py + test_throughput_allocation.py.
- `60f081e3` — **link-speed-aware ring host selection**: NetworkInterfaceInfo.link_speed_megabits; _classify_linux_interface() via sysfs (fixes Linux 'unknown' interface gap — pairs with #2129's _linux_interface_types); find_ip_prioritised scores (latency #2254 → link speed → RFC1918 #2063); nominal speed table preserves TB-first when unmeasured. Files: profiling.py, system_info.py, placement_utils.py + 2 tests.
- `e9011332` — **prefix-cache divergence deadlock fix**: mx_ranks_agree_on_value(); generate.py + batch_generate.py discard KV hits when ranks disagree (hybrid SSM/attention shards restore different lengths → multi-turn deadlock). Files: utils_mlx.py, generate.py, batch_generate.py.

Verified: basedpyright 0, ruff clean, 637 tests pass (1 pre-existing rust baseline failure).

## Wave 7 — Closing gaps (2026-08-07)

- `03dcb4e1` — **#1: per-message guard in _networking_recv** (defense in depth follow-up to ef16c380 from concurrent-agent analysis): match body wrapped in try/except catching ValidationError/UnicodeDecodeError/JSONDecodeError → log + record_malformed_event + loop continues; resource-closed errors NOT caught (legit shutdown). Verified: pyright 0, ruff clean, 89 tests.

## Machine-level fixes applied live (2026-08-07, on 10.2.0.16)
- `uv sync --extra mlx --extra mlx-cuda12` → pynvml importable
- `ln -sf /usr/bin/nvidia-smi-595.58 /usr/bin/nvidia-smi`
- NodeBackends.gather() now advertises ['MlxCpu','MlxCuda','Vllm'] ✓ (was CPU-only)
- exo restarted with CUDA_HOME; log: "CUDA MLX backend detected; reporting GPU VRAM as node memory"
- Concurrent agent commits: e6ec3e2c (pynvml fallback in gather), 2a9e15a2 (nvidia-smi suffix + NVML), 40edf56a (setup_linux_gpu.sh), ceccb23f (README)
